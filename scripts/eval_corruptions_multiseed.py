"""FP32 corruption robustness across all 5 seeds, for the statistical-rigor extension.

Same corruption sweep as eval_corruptions.py (which stays as-is, producing the seed-42
headline numbers) but looped over every seed so mean/std/CI can be computed downstream by
scripts/compute_statistics.py.
"""

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

SEEDS = [42, 123, 2024, 7, 999]

ARCHS = {
    "baseline_cnn": {
        "model_fn": lambda: BaselineCNN(num_classes=43),
        "mean": NORM_MEAN,
        "std": NORM_STD,
    },
    "mobilenet_transfer": {
        "model_fn": lambda: build_mobilenet(num_classes=43),
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
}


def evaluate(model, test_df, device, mean, std, corruption_fn=None):
    transform = get_transform(mean=mean, std=std, corruption_fn=corruption_fn)
    ds = GTSRBDataset(test_df, transform=transform)
    # num_workers=0: forked DataLoader workers deadlock against the MPS backend on this
    # machine once a long-running multi-seed loop keeps the device context alive across
    # many models (single-seed eval_corruptions.py never hit this at n=1 model load).
    preds, labels = predict(model, ds, device, num_workers=0)
    s = summarize(preds, labels)
    return {"accuracy": s["accuracy"], "macro_f1": s["macro_f1"]}


def main():
    device = get_device()
    set_seed(42)
    test_df = load_test_dataframe(RAW_DIR)

    results = {}
    for arch_name, cfg in ARCHS.items():
        results[arch_name] = {}
        for seed in SEEDS:
            model = cfg["model_fn"]()
            model.load_state_dict(torch.load(CKPT_DIR / f"{arch_name}_seed{seed}.pt", map_location=device))
            model = model.to(device).eval()

            seed_results = {"clean": evaluate(model, test_df, device, cfg["mean"], cfg["std"])}
            for corr_name, factory in CORRUPTIONS.items():
                seed_results[corr_name] = []
                for severity in SEVERITIES:
                    metrics = evaluate(
                        model, test_df, device, cfg["mean"], cfg["std"], corruption_fn=factory(severity)
                    )
                    metrics["severity"] = severity
                    seed_results[corr_name].append(metrics)
            results[arch_name][str(seed)] = seed_results
            print(f"{arch_name} seed{seed}: clean={seed_results['clean']}")

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "corruption_results_multiseed_fp32.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
