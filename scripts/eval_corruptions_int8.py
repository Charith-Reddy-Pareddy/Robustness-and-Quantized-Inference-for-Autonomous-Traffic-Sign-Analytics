"""Re-run the Day 4 corruption robustness tests directly on the INT8 ONNX models.

The exported ONNX graphs already have normalization baked in (see src/models/wrapper.py),
so this uses the same pixel-space [0,1] transform for every architecture — no per-model
mean/std needed here, unlike the PyTorch-side scripts.
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
ONNX_DIR = ROOT / "onnx"
REPORTS_DIR = ROOT / "reports"

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
        onnx_path = ONNX_DIR / f"{arch_name}_int8.onnx"
        arch_results = {"clean": evaluate(onnx_path, test_df)}
        print(f"{arch_name} (INT8) clean: {arch_results['clean']}")

        for corr_name, factory in CORRUPTIONS.items():
            arch_results[corr_name] = []
            for severity in SEVERITIES:
                metrics = evaluate(onnx_path, test_df, corruption_fn=factory(severity))
                metrics["severity"] = severity
                arch_results[corr_name].append(metrics)
                print(f"{arch_name} (INT8) {corr_name} sev={severity}: {metrics}")

        results[arch_name] = arch_results

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "corruption_results_int8.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
