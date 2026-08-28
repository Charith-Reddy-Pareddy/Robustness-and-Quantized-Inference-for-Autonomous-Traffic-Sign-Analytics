"""Second generalization check: evaluate the GTSRB-trained models (FP32 and INT8) on
BelgiumTSC, restricted to the 18 GTSRB classes with a confident semantic match (see
src/data/belgium_mapping.py). Tests whether the Mapillary+DFG generalization finding
(both models generalize far worse than GTSRB numbers suggest, but the FP32-vs-INT8 gap
stays small) holds on a second, independent out-of-distribution dataset -- a different
country (Belgium vs. Mapillary+DFG's Africa-region crops), different capture setup
(roof-mounted van cameras vs. crowdsourced street-level imagery), same Vienna Convention
sign family as GTSRB's Germany.
"""

import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.belgium_ingest import load_belgium_dataframe
from src.data.belgium_mapping import GTSRB_TO_BELGIUM
from src.data.dataset import GTSRBDataset
from src.data.ingest import load_test_dataframe
from src.data.transforms import get_transform
from src.models.evaluate import predict, summarize
from src.models.onnx_infer import predict_onnx
from src.models.registry import archs_with_ckpt
from src.models.train import get_device

RAW_DIR = ROOT / "data" / "raw"
BELGIUM_DIR = ROOT / "data" / "belgium"
CKPT_DIR = ROOT / "checkpoints"
ONNX_DIR = ROOT / "onnx"
REPORTS_DIR = ROOT / "reports"

MAPPED_CLASSES = sorted(GTSRB_TO_BELGIUM.keys())

ARCHS = archs_with_ckpt()


def restricted_summary(preds, labels):
    accuracy = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro", labels=MAPPED_CLASSES))
    return accuracy, macro_f1


def main():
    device = get_device()
    print(f"Using device: {device}")

    belgium_df = load_belgium_dataframe(BELGIUM_DIR)
    print(f"BelgiumTSC generalization set: {len(belgium_df)} images, {belgium_df['ClassId'].nunique()} classes")

    gtsrb_test_df = load_test_dataframe(RAW_DIR)
    gtsrb_test_restricted = gtsrb_test_df[gtsrb_test_df["ClassId"].isin(MAPPED_CLASSES)].reset_index(drop=True)
    print(f"GTSRB test set restricted to the same {len(MAPPED_CLASSES)} classes: {len(gtsrb_test_restricted)} images")

    results = {}
    for arch_name, cfg in ARCHS.items():
        transform = get_transform(mean=cfg["mean"], std=cfg["std"])
        ds = GTSRBDataset(belgium_df, transform=transform, crop_to_roi=True)

        base_model = cfg["model_fn"]()
        base_model.load_state_dict(torch.load(CKPT_DIR / cfg["ckpt"], map_location=device))
        base_model = base_model.to(device).eval()
        preds, labels = predict(base_model, ds, device)
        fp32_acc, fp32_f1 = restricted_summary(preds, labels)
        print(f"{arch_name} FP32 on BelgiumTSC: acc={fp32_acc:.4f} macro_f1={fp32_f1:.4f}")

        gtsrb_ds = GTSRBDataset(gtsrb_test_restricted, transform=transform, crop_to_roi=True)
        gtsrb_preds, gtsrb_labels = predict(base_model, gtsrb_ds, device)
        gtsrb_acc, gtsrb_f1 = restricted_summary(gtsrb_preds, gtsrb_labels)
        print(f"{arch_name} FP32 on GTSRB test (same {len(MAPPED_CLASSES)} classes): acc={gtsrb_acc:.4f} macro_f1={gtsrb_f1:.4f}")

        pixel_transform = get_transform(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        onnx_ds = GTSRBDataset(belgium_df, transform=pixel_transform, crop_to_roi=True)
        int8_preds, int8_labels = predict_onnx(ONNX_DIR / f"{arch_name}_int8.onnx", onnx_ds, num_workers=0)
        int8_acc, int8_f1 = restricted_summary(int8_preds, int8_labels)
        print(f"{arch_name} INT8 on BelgiumTSC: acc={int8_acc:.4f} macro_f1={int8_f1:.4f}")

        report = summarize(preds, labels)["report"]
        per_class_f1 = {str(c): report.get(str(c), {}).get("f1-score", 0.0) for c in MAPPED_CLASSES}
        per_class_n = {str(c): int((labels == c).sum()) for c in MAPPED_CLASSES}

        results[arch_name] = {
            "gtsrb_test_accuracy_restricted": gtsrb_acc,
            "gtsrb_test_macro_f1_restricted": gtsrb_f1,
            "fp32_accuracy": fp32_acc,
            "fp32_macro_f1": fp32_f1,
            "int8_accuracy": int8_acc,
            "int8_macro_f1": int8_f1,
            "per_class_f1_fp32": per_class_f1,
            "per_class_n": per_class_n,
        }

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "belgium_generalization_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
