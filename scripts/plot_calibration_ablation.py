"""Clean accuracy and severity-4 blur accuracy vs. INT8 calibration-set size."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "reports"

CALIBRATION_SIZES = [50, 100, 200, 500, 1000, 2000]
ARCH_LABELS = {"baseline_cnn": "Baseline CNN", "mobilenet_transfer": "MobileNetV2 (transfer)"}
ARCH_COLORS = {"baseline_cnn": "#4C72B0", "mobilenet_transfer": "#DD8452"}


def main():
    results = json.loads((REPORTS_DIR / "calibration_ablation_results.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for arch_name, arch_results in results.items():
        clean = [arch_results[str(n)]["clean"]["accuracy"] * 100 for n in CALIBRATION_SIZES]
        blur4 = [
            next(m["accuracy"] for m in arch_results[str(n)]["blur"] if m["severity"] == 4) * 100
            for n in CALIBRATION_SIZES
        ]
        axes[0].plot(CALIBRATION_SIZES, clean, marker="o", label=ARCH_LABELS[arch_name], color=ARCH_COLORS[arch_name])
        axes[1].plot(CALIBRATION_SIZES, blur4, marker="o", label=ARCH_LABELS[arch_name], color=ARCH_COLORS[arch_name])

    axes[0].set_title("Clean accuracy")
    axes[1].set_title("Blur, severity 4")
    for ax in axes:
        ax.set_xlabel("Calibration set size")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("INT8 accuracy vs. calibration-set size (model size is constant at every size)")
    fig.tight_layout()
    out_path = REPORTS_DIR / "calibration_ablation.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
