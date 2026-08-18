"""Evaluate both trained models (seed 42) on GTSRB Test set under clean + corrupted conditions."""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import GTSRBDataset
from src.data.ingest import load_test_dataframe
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, NORM_MEAN, NORM_STD, get_transform
from src.models.baseline_cnn import BaselineCNN
from src.models.evaluate import predict, summarize
from src.models.train import get_device, set_seed
from src.models.transfer_model import build_mobilenet
from src.robustness.corruptions import CORRUPTIONS, SEVERITIES

RAW_DIR = ROOT / "data" / "raw"
CKPT_DIR = ROOT / "checkpoints"
REPORTS_DIR = ROOT / "reports"

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


def evaluate(model, test_df, device, mean, std, corruption_fn=None):
    transform = get_transform(mean=mean, std=std, corruption_fn=corruption_fn)
    ds = GTSRBDataset(test_df, transform=transform)
    preds, labels = predict(model, ds, device)
    s = summarize(preds, labels)
    return {"accuracy": s["accuracy"], "macro_f1": s["macro_f1"]}


def main():
    device = get_device()
    set_seed(42)
    test_df = load_test_dataframe(RAW_DIR)

    results = {}
    for arch_name, cfg in ARCHS.items():
        model = cfg["model_fn"]()
        model.load_state_dict(torch.load(CKPT_DIR / cfg["ckpt"], map_location=device))
        model = model.to(device).eval()

        arch_results = {"clean": evaluate(model, test_df, device, cfg["mean"], cfg["std"])}
        print(f"{arch_name} clean: {arch_results['clean']}")

        for corr_name, factory in CORRUPTIONS.items():
            arch_results[corr_name] = []
            for severity in SEVERITIES:
                metrics = evaluate(
                    model, test_df, device, cfg["mean"], cfg["std"], corruption_fn=factory(severity)
                )
                metrics["severity"] = severity
                arch_results[corr_name].append(metrics)
                print(f"{arch_name} {corr_name} sev={severity}: {metrics}")

        results[arch_name] = arch_results

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "corruption_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
