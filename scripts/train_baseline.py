"""Train the baseline CNN on GTSRB and evaluate on the held-out Test set."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import GTSRBDataset
from src.data.ingest import load_test_dataframe, load_train_dataframe, track_aware_split
from src.data.transforms import get_transform
from src.models.baseline_cnn import BaselineCNN
from src.models.evaluate import predict, summarize
from src.models.train import TrainConfig, get_device, train_model

RAW_DIR = ROOT / "data" / "raw"
CKPT_DIR = ROOT / "checkpoints"
REPORTS_DIR = ROOT / "reports"


def main():
    device = get_device()
    print(f"Using device: {device}")

    transform = get_transform()
    train_df = load_train_dataframe(RAW_DIR)
    train_split_df, val_split_df = track_aware_split(train_df, val_fraction=0.15, seed=42)
    test_df = load_test_dataframe(RAW_DIR)

    train_ds = GTSRBDataset(train_split_df, transform=transform)
    val_ds = GTSRBDataset(val_split_df, transform=transform)
    test_ds = GTSRBDataset(test_df, transform=transform)

    model = BaselineCNN(num_classes=43)
    config = TrainConfig(epochs=12, lr=1e-3, batch_size=128, seed=42, patience=3)
    ckpt_path = CKPT_DIR / "baseline_cnn_seed42.pt"
    history = train_model(model, train_ds, val_ds, device, config, ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    preds, labels = predict(model, test_ds, device)
    results = summarize(preds, labels)
    print(f"Test accuracy: {results['accuracy']:.4f}  Test macro-F1: {results['macro_f1']:.4f}")

    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "baseline_cnn_metrics.json", "w") as f:
        json.dump(
            {
                "history": history,
                "test_accuracy": results["accuracy"],
                "test_macro_f1": results["macro_f1"],
                "per_class_report": results["report"],
            },
            f,
            indent=2,
        )

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(results["confusion_matrix"], ax=ax, cmap="Blues", cbar=True)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Baseline CNN — Test set confusion matrix")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "baseline_cnn_confusion_matrix.png", dpi=120)
    print(f"Saved reports to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
