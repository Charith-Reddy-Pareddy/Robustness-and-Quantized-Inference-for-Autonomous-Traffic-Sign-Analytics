"""Evaluate clean accuracy + corruption robustness for each calibration-size INT8 model
produced by quantize_calibration_ablation.py, plus latency/size at each size (sanity
check that these are calibration-independent, as the fixed-weight-precision quantization
scheme implies).
"""

import json
import sys
import time
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
ONNX_DIR = ROOT / "onnx" / "calibration_ablation"
REPORTS_DIR = ROOT / "reports"

CALIBRATION_SIZES = [50, 100, 200, 500, 1000, 2000]
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
        for n in CALIBRATION_SIZES:
            onnx_path = ONNX_DIR / f"{arch_name}_calib{n}_int8.onnx"
            size_kb = onnx_path.stat().st_size / 1024

            size_results = {"model_size_kb": size_kb, "clean": evaluate(onnx_path, test_df)}
            for corr_name, factory in CORRUPTIONS.items():
                size_results[corr_name] = []
                for severity in SEVERITIES:
                    metrics = evaluate(onnx_path, test_df, corruption_fn=factory(severity))
                    metrics["severity"] = severity
                    size_results[corr_name].append(metrics)
            results[arch_name][str(n)] = size_results
            print(f"{arch_name} calib={n}: clean={size_results['clean']}, size={size_kb:.1f}KB")

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "calibration_ablation_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
