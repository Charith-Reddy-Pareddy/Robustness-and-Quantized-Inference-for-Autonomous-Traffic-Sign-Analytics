"""Plots for the Mapillary generalization check: overall accuracy comparison (GTSRB test
set vs. Mapillary, FP32 vs INT8) and per-class F1 on Mapillary."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.mapillary_mapping import GTSRB_CLASS_NAMES

REPORTS_DIR = ROOT / "reports"

ARCH_LABELS = {"baseline_cnn": "Baseline CNN", "mobilenet_transfer": "MobileNetV2 (transfer)"}
ARCH_COLORS = {"baseline_cnn": "#4C72B0", "mobilenet_transfer": "#DD8452"}


def main():
    results = json.loads((REPORTS_DIR / "mapillary_generalization_results.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    archs = list(results.keys())
    x = range(len(archs))
    width = 0.25
    gtsrb_acc = [results[a]["gtsrb_test_accuracy_restricted"] for a in archs]
    fp32_acc = [results[a]["fp32_accuracy"] for a in archs]
    int8_acc = [results[a]["int8_accuracy"] for a in archs]
    ax.bar([i - width for i in x], gtsrb_acc, width, label="GTSRB test (FP32)", color="#888888")
    ax.bar(list(x), fp32_acc, width, label="Mapillary (FP32)", color="#4C72B0")
    ax.bar([i + width for i in x], int8_acc, width, label="Mapillary (INT8)", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels([ARCH_LABELS[a] for a in archs])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Accuracy: GTSRB test set vs. Mapillary generalization set")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    arch = "mobilenet_transfer"
    per_class = results[arch]["per_class_f1_fp32"]
    items = sorted(per_class.items(), key=lambda kv: kv[1])
    labels = [GTSRB_CLASS_NAMES[int(c)] for c, _ in items]
    values = [v for _, v in items]
    ax.barh(labels, values, color=ARCH_COLORS[arch])
    ax.set_xlabel("F1 score")
    ax.set_title(f"{ARCH_LABELS[arch]}: per-class F1 on Mapillary (FP32)")
    ax.set_xlim(0, 1.0)

    fig.tight_layout()
    out_path = REPORTS_DIR / "mapillary_generalization.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
