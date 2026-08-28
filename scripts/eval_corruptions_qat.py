"""Corruption robustness for the QAT-converted (real INT8, PyTorch/qnnpack) models,
seed 42, comparable to eval_corruptions_int8.py's PTQ (ONNX Runtime) numbers.
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import GTSRBDataset
from src.data.ingest import load_test_dataframe
from src.data.transforms import get_transform
from src.models.evaluate import predict, summarize
from src.models.qat import load_converted
from src.models.registry import ARCH_SPECS
from src.robustness.corruptions import CORRUPTIONS, SEVERITIES

RAW_DIR = ROOT / "data" / "raw"
CKPT_DIR = ROOT / "checkpoints"
REPORTS_DIR = ROOT / "reports"

ARCHS = {name: {"mean": spec["mean"], "std": spec["std"]} for name, spec in ARCH_SPECS.items()}


def evaluate(model, test_df, mean, std, corruption_fn=None):
    transform = get_transform(mean=mean, std=std, corruption_fn=corruption_fn)
    ds = GTSRBDataset(test_df, transform=transform)
    # QAT-converted (real quantized) modules only run on CPU; num_workers=0 to avoid the
    # same MPS-adjacent forked-dataloader issue hit in eval_corruptions_multiseed.py.
    preds, labels = predict(model, ds, torch.device("cpu"), num_workers=0)
    s = summarize(preds, labels)
    return {"accuracy": s["accuracy"], "macro_f1": s["macro_f1"]}


def main():
    test_df = load_test_dataframe(RAW_DIR)
    results = {}
    for arch_name, cfg in ARCHS.items():
        model = load_converted(arch_name, CKPT_DIR / f"{arch_name}_qat_seed42_converted.pt")
        model.eval()

        arch_results = {"clean": evaluate(model, test_df, cfg["mean"], cfg["std"])}
        print(f"{arch_name} (QAT INT8) clean: {arch_results['clean']}")
        for corr_name, factory in CORRUPTIONS.items():
            arch_results[corr_name] = []
            for severity in SEVERITIES:
                metrics = evaluate(model, test_df, cfg["mean"], cfg["std"], corruption_fn=factory(severity))
                metrics["severity"] = severity
                arch_results[corr_name].append(metrics)
                print(f"{arch_name} (QAT INT8) {corr_name} sev={severity}: {metrics}")

        results[arch_name] = arch_results

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "corruption_results_qat.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
