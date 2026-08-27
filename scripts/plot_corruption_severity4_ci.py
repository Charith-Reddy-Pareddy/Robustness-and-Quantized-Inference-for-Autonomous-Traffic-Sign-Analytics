"""Multi-seed FP32 vs. INT8 accuracy at corruption severity 4, with 95% CI error bars."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "reports"

CORRUPTIONS = ["blur", "noise", "rotation", "brightness_contrast"]
ARCH_LABELS = {"baseline_cnn": "Baseline CNN", "mobilenet_transfer": "MobileNetV2 (transfer)"}


def main():
    stats = json.loads((REPORTS_DIR / "statistics.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    x = np.arange(len(CORRUPTIONS))
    width = 0.35

    for ax, arch_name in zip(axes, ["baseline_cnn", "mobilenet_transfer"]):
        fp32_means, fp32_errs, int8_means, int8_errs = [], [], [], []
        for corr in CORRUPTIONS:
            entry = stats["corruption"][arch_name][corr]["4"]
            fp32_means.append(entry["fp32"]["mean"] * 100)
            fp32_errs.append((entry["fp32"]["mean"] - entry["fp32"]["ci_low"]) * 100)
            int8_means.append(entry["int8"]["mean"] * 100)
            int8_errs.append((entry["int8"]["mean"] - entry["int8"]["ci_low"]) * 100)

        ax.bar(x - width / 2, fp32_means, width, yerr=fp32_errs, capsize=4, label="FP32", color="#4C72B0")
        ax.bar(x + width / 2, int8_means, width, yerr=int8_errs, capsize=4, label="INT8", color="#DD8452")
        ax.set_title(ARCH_LABELS[arch_name])
        ax.set_xticks(x)
        ax.set_xticklabels(CORRUPTIONS, rotation=20, ha="right")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("Severity-4 corruption accuracy: FP32 vs. INT8, mean +/- 95% CI over 5 seeds")
    fig.tight_layout()
    out_path = REPORTS_DIR / "corruption_severity4_ci.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
