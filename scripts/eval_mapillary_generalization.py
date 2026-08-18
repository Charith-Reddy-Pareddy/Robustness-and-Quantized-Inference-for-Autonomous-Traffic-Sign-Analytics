"""Generalization check: evaluate the GTSRB-trained models (FP32 and INT8) on the
Mapillary+DFG dataset, restricted to the 23 GTSRB classes with a confident semantic match
(see src/data/mapillary_mapping.py). Tests whether the FP32-vs-INT8 robustness findings
are GTSRB-specific or hold on a different, more diverse real-world image distribution.
"""

import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import GTSRBDataset
from src.data.ingest import load_test_dataframe
from src.data.mapillary_ingest import load_mapillary_dataframe
from src.data.mapillary_mapping import GTSRB_TO_MAPILLARY
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, NORM_MEAN, NORM_STD, get_transform
from src.models.baseline_cnn import BaselineCNN
from src.models.evaluate import predict, summarize
from src.models.onnx_infer import predict_onnx
from src.models.train import get_device
from src.models.transfer_model import build_mobilenet

RAW_DIR = ROOT / "data" / "raw"
MAPILLARY_DIR = ROOT / "data" / "mapillary"
CKPT_DIR = ROOT / "checkpoints"
ONNX_DIR = ROOT / "onnx"
REPORTS_DIR = ROOT / "reports"

SAMPLES_PER_CLASS = 150
MAPPED_CLASSES = sorted(GTSRB_TO_MAPILLARY.keys())

ARCHS = {
    "baseline_cnn": {
        "model_fn": lambda: BaselineCNN(num_classes=43),
        "mean": NORM_MEAN,
        "std": NORM_STD,
        "ckpt": "baseline_cnn_seed42.pt",
    },
    "mobilenet_transfer": {
        "model_fn": lambda: build_mobilenet(num_classes=43),
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "ckpt": "mobilenet_transfer_seed42.pt",
    },
}


def restricted_summary(preds, labels):
    """Accuracy/macro-F1 computed only over the 23 mapped classes (the model can still
    predict any of the 43 GTSRB classes -- predicting an unmapped class is just wrong,
    same as any other misclassification)."""
    accuracy = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro", labels=MAPPED_CLASSES))
    return accuracy, macro_f1


def main():
    device = get_device()
    print(f"Using device: {device}")

    mapillary_df = load_mapillary_dataframe(MAPILLARY_DIR, samples_per_class=SAMPLES_PER_CLASS, seed=42)
    print(f"Mapillary generalization set: {len(mapillary_df)} images, {mapillary_df['ClassId'].nunique()} classes")

    gtsrb_test_df = load_test_dataframe(RAW_DIR)
    gtsrb_test_restricted = gtsrb_test_df[gtsrb_test_df["ClassId"].isin(MAPPED_CLASSES)].reset_index(drop=True)
    print(f"GTSRB test set restricted to the same {len(MAPPED_CLASSES)} classes: {len(gtsrb_test_restricted)} images")

    results = {}
    for arch_name, cfg in ARCHS.items():
        transform = get_transform(mean=cfg["mean"], std=cfg["std"])
        ds = GTSRBDataset(mapillary_df, transform=transform, crop_to_roi=False)

        base_model = cfg["model_fn"]()
        base_model.load_state_dict(torch.load(CKPT_DIR / cfg["ckpt"], map_location=device))
        base_model = base_model.to(device).eval()
        preds, labels = predict(base_model, ds, device)
        fp32_acc, fp32_f1 = restricted_summary(preds, labels)
        print(f"{arch_name} FP32 on Mapillary: acc={fp32_acc:.4f} macro_f1={fp32_f1:.4f}")

        gtsrb_ds = GTSRBDataset(gtsrb_test_restricted, transform=transform, crop_to_roi=True)
        gtsrb_preds, gtsrb_labels = predict(base_model, gtsrb_ds, device)
        gtsrb_acc, gtsrb_f1 = restricted_summary(gtsrb_preds, gtsrb_labels)
        print(f"{arch_name} FP32 on GTSRB test (same {len(MAPPED_CLASSES)} classes): acc={gtsrb_acc:.4f} macro_f1={gtsrb_f1:.4f}")

        pixel_transform = get_transform(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        onnx_ds = GTSRBDataset(mapillary_df, transform=pixel_transform, crop_to_roi=False)
        int8_preds, int8_labels = predict_onnx(ONNX_DIR / f"{arch_name}_int8.onnx", onnx_ds, num_workers=0)
        int8_acc, int8_f1 = restricted_summary(int8_preds, int8_labels)
        print(f"{arch_name} INT8 on Mapillary: acc={int8_acc:.4f} macro_f1={int8_f1:.4f}")

        report = summarize(preds, labels)["report"]
        per_class_f1 = {str(c): report.get(str(c), {}).get("f1-score", 0.0) for c in MAPPED_CLASSES}

        results[arch_name] = {
            "gtsrb_test_accuracy_restricted": gtsrb_acc,
            "gtsrb_test_macro_f1_restricted": gtsrb_f1,
            "fp32_accuracy": fp32_acc,
            "fp32_macro_f1": fp32_f1,
            "int8_accuracy": int8_acc,
            "int8_macro_f1": int8_f1,
            "per_class_f1_fp32": per_class_f1,
        }

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "mapillary_generalization_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
