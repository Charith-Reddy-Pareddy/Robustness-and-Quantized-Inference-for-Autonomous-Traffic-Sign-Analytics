"""Accuracy-vs-corruption-severity curves — required robustness-report deliverable."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "reports"

ARCH_LABELS = {"baseline_cnn": "Baseline CNN", "mobilenet_transfer": "MobileNetV2 (transfer)"}
ARCH_COLORS = {"baseline_cnn": "#4C72B0", "mobilenet_transfer": "#DD8452"}
CORRUPTION_TITLES = {
    "blur": "Gaussian blur",
    "noise": "Gaussian noise",
    "rotation": "Rotation",
    "brightness_contrast": "Brightness/contrast shift",
}


def main():
    results = json.loads((REPORTS_DIR / "corruption_results.json").read_text())
    corruption_names = list(CORRUPTION_TITLES.keys())

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    for ax, corr_name in zip(axes.flat, corruption_names):
        for arch_name, arch_results in results.items():
            severities = [0] + [m["severity"] for m in arch_results[corr_name]]
            accuracies = [arch_results["clean"]["accuracy"]] + [
                m["accuracy"] for m in arch_results[corr_name]
            ]
            ax.plot(
                severities,
                accuracies,
                marker="o",
                label=ARCH_LABELS[arch_name],
                color=ARCH_COLORS[arch_name],
            )
        ax.set_title(CORRUPTION_TITLES[corr_name])
        ax.set_xlabel("Severity (0 = clean)")
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Accuracy vs. corruption severity")
    fig.tight_layout()
    out_path = REPORTS_DIR / "corruption_curves.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
