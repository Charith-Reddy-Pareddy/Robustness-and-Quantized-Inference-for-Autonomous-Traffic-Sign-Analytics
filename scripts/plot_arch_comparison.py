"""Bar chart comparing baseline CNN vs MobileNetV2 transfer model, mean +/- std over seeds."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "reports"


def main():
    summary = json.loads((REPORTS_DIR / "multi_seed_summary.json").read_text())
    archs = list(summary.keys())
    labels = ["Baseline CNN" if a == "baseline_cnn" else "MobileNetV2 (transfer)" for a in archs]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric_key, title in zip(
        axes, ["accuracy", "macro_f1"], ["Test accuracy", "Test macro-F1"]
    ):
        means = [summary[a][f"{metric_key}_mean"] for a in archs]
        stds = [summary[a][f"{metric_key}_std"] for a in archs]
        ax.bar(labels, means, yerr=stds, capsize=6, color=["#4C72B0", "#DD8452"])
        ax.set_title(f"{title} (mean +/- std, n=3 seeds)")
        ax.set_ylim(0, 1.0)
        for i, m in enumerate(means):
            ax.text(i, m + 0.02, f"{m:.3f}", ha="center")

    fig.tight_layout()
    out_path = REPORTS_DIR / "arch_comparison.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
