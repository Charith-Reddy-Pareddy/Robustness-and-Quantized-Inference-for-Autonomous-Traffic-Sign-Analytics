"""INT8 corruption robustness across all 5 seeds, for the statistical-rigor extension.

Requires scripts/quantize_models_multiseed.py to have been run first (produces
onnx/multiseed/{arch}_seed{seed}_int8.onnx for every seed).
"""

import json
import sys
from pathlib import Path

from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import GTSRBDataset
from src.data.ingest import load_test_dataframe
from src.data.transforms import get_transform
from src.models.onnx_infer import predict_onnx
from src.robustness.corruptions import CORRUPTIONS, SEVERITIES

RAW_DIR = ROOT / "data" / "raw"
ONNX_DIR = ROOT / "onnx" / "multiseed"
REPORTS_DIR = ROOT / "reports"

SEEDS = [42, 123, 2024, 7, 999]
ARCHS = ["baseline_cnn", "mobilenet_transfer"]


def evaluate(onnx_path, test_df, corruption_fn=None):
    transform = get_transform(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], corruption_fn=corruption_fn)
    ds = GTSRBDataset(test_df, transform=transform)
    preds, labels = predict_onnx(onnx_path, ds, batch_size=128, num_workers=0)
    accuracy = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def main():
    test_df = load_test_dataframe(RAW_DIR)
    results = {}
    for arch_name in ARCHS:
        results[arch_name] = {}
        for seed in SEEDS:
            onnx_path = ONNX_DIR / f"{arch_name}_seed{seed}_int8.onnx"
            seed_results = {"clean": evaluate(onnx_path, test_df)}
            for corr_name, factory in CORRUPTIONS.items():
                seed_results[corr_name] = []
                for severity in SEVERITIES:
                    metrics = evaluate(onnx_path, test_df, corruption_fn=factory(severity))
                    metrics["severity"] = severity
                    seed_results[corr_name].append(metrics)
            results[arch_name][str(seed)] = seed_results
            print(f"{arch_name} (INT8) seed{seed}: clean={seed_results['clean']}")

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "corruption_results_multiseed_int8.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
