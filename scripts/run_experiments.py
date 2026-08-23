"""Train each architecture across multiple seeds and report mean +/- std on test metrics."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import GTSRBDataset
from src.data.ingest import load_test_dataframe, load_train_dataframe, track_aware_split
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, NORM_MEAN, NORM_STD, get_transform
from src.models.baseline_cnn import BaselineCNN
from src.models.evaluate import predict, summarize
from src.models.train import TrainConfig, get_device, train_model
from src.models.transfer_model import build_mobilenet

RAW_DIR = ROOT / "data" / "raw"
CKPT_DIR = ROOT / "checkpoints"
REPORTS_DIR = ROOT / "reports"

SEEDS = [42, 123, 2024, 7, 999]

ARCHS = {
    "baseline_cnn": {
        "model_fn": lambda: BaselineCNN(num_classes=43),
        "mean": NORM_MEAN,
        "std": NORM_STD,
        "lr": 1e-3,
        "epochs": 12,
    },
    "mobilenet_transfer": {
        "model_fn": lambda: build_mobilenet(num_classes=43),
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "lr": 1e-4,
        "epochs": 10,
    },
}


def run_one(arch_name: str, arch_cfg: dict, seed: int, device) -> dict:
    ckpt_path = CKPT_DIR / f"{arch_name}_seed{seed}.pt"
    metrics_path = REPORTS_DIR / f"{arch_name}_seed{seed}_metrics.json"
    if metrics_path.exists() and ckpt_path.exists():
        print(f"[skip] {arch_name} seed={seed} already has results at {metrics_path}")
        return json.loads(metrics_path.read_text())

    transform = get_transform(mean=arch_cfg["mean"], std=arch_cfg["std"])
    train_df = load_train_dataframe(RAW_DIR)
    train_split_df, val_split_df = track_aware_split(train_df, val_fraction=0.15, seed=42)
    test_df = load_test_dataframe(RAW_DIR)

    train_ds = GTSRBDataset(train_split_df, transform=transform)
    val_ds = GTSRBDataset(val_split_df, transform=transform)
    test_ds = GTSRBDataset(test_df, transform=transform)

    model = arch_cfg["model_fn"]()
    config = TrainConfig(epochs=arch_cfg["epochs"], lr=arch_cfg["lr"], batch_size=128, seed=seed, patience=3)
    print(f"=== Training {arch_name} seed={seed} ===")
    history = train_model(model, train_ds, val_ds, device, config, ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    preds, labels = predict(model, test_ds, device)
    results = summarize(preds, labels)

    metrics = {
        "arch": arch_name,
        "seed": seed,
        "test_accuracy": results["accuracy"],
        "test_macro_f1": results["macro_f1"],
        "history": history,
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    device = get_device()
    print(f"Using device: {device}")

    all_results = {}
    for arch_name, arch_cfg in ARCHS.items():
        per_seed = [run_one(arch_name, arch_cfg, seed, device) for seed in SEEDS]
        accs = np.array([m["test_accuracy"] for m in per_seed])
        f1s = np.array([m["test_macro_f1"] for m in per_seed])
        all_results[arch_name] = {
            "per_seed": [{"seed": m["seed"], "accuracy": m["test_accuracy"], "macro_f1": m["test_macro_f1"]} for m in per_seed],
            "accuracy_mean": float(accs.mean()),
            "accuracy_std": float(accs.std()),
            "macro_f1_mean": float(f1s.mean()),
            "macro_f1_std": float(f1s.std()),
        }
        print(
            f"{arch_name}: acc={accs.mean():.4f}+/-{accs.std():.4f}  "
            f"macro_f1={f1s.mean():.4f}+/-{f1s.std():.4f}"
        )

    summary_path = REPORTS_DIR / "multi_seed_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
