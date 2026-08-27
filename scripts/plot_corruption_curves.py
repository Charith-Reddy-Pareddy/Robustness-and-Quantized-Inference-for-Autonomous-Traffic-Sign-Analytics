"""Accuracy-vs-corruption-severity curves — required robustness-report deliverable."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.robustness.corruptions import CORRUPTION_CATEGORIES

REPORTS_DIR = ROOT / "reports"

ARCH_LABELS = {"baseline_cnn": "Baseline CNN", "mobilenet_transfer": "MobileNetV2 (transfer)"}
ARCH_COLORS = {"baseline_cnn": "#4C72B0", "mobilenet_transfer": "#DD8452"}
CORRUPTION_TITLES = {
    "blur": "Gaussian blur",
    "noise": "Gaussian noise",
    "rotation": "Rotation",
    "brightness_contrast": "Brightness/contrast shift",
    "jpeg_compression": "JPEG compression",
    "motion_blur": "Motion blur",
    "fog": "Fog",
    "rain": "Rain",
    "shadow": "Shadow",
    "perspective": "Perspective warp",
}
# Flattened in category order (noise, blur, geometric, photometric, environmental) so
# related corruptions sit next to each other in the grid.
CORRUPTION_ORDER = [name for names in CORRUPTION_CATEGORIES.values() for name in names]


def main():
    results = json.loads((REPORTS_DIR / "corruption_results.json").read_text())

    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharey=True)
    for ax, corr_name in zip(axes.flat, CORRUPTION_ORDER):
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
        ax.set_title(CORRUPTION_TITLES[corr_name], fontsize=10)
        ax.set_xlabel("Severity (0 = clean)")
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Accuracy vs. corruption severity, by category (noise / blur / geometric / photometric / environmental)")
    fig.tight_layout()
    out_path = REPORTS_DIR / "corruption_curves.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
