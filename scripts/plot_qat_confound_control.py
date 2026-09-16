"""Plot the QAT training-epochs confound control: FP32_original vs. FP32_extra_training
vs. QAT, on clean accuracy, severity-4 corruption, and white-box PGD.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "reports"
ARCH_LABELS = {"baseline_cnn": "Baseline CNN", "mobilenet_transfer": "MobileNetV2 (transfer)"}
VARIANT_LABELS = {
    "fp32_original": "FP32 (original)",
    "fp32_extra_training": "FP32 (+ extra training)",
    "qat": "QAT",
}
VARIANT_COLORS = {"fp32_original": "#4C72B0", "fp32_extra_training": "#DD8452", "qat": "#55A868"}


def main():
    results = json.loads((REPORTS_DIR / "qat_confound_control.json").read_text())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, arch in enumerate(ARCH_LABELS):
        bar_ax, line_ax = axes[row]
        arch_results = results[arch]

        metrics = ["clean_accuracy", "corruption_severity4"]
        metric_labels = ["Clean accuracy", "Corruption (sev. 4, mean)"]
        x = np.arange(len(metrics))
        width = 0.26
        for i, variant in enumerate(VARIANT_LABELS):
            values = [
                arch_results[variant]["clean_accuracy"],
                arch_results[variant]["corruption_severity4"]["mean_accuracy"],
            ]
            bar_ax.bar(x + (i - 1) * width, values, width, label=VARIANT_LABELS[variant], color=VARIANT_COLORS[variant])
        bar_ax.set_xticks(x, metric_labels)
        bar_ax.set_ylim(0, 1.0)
        bar_ax.set_ylabel("Accuracy")
        bar_ax.set_title(f"{ARCH_LABELS[arch]}: clean + corruption")
        bar_ax.grid(alpha=0.3, axis="y")
        bar_ax.legend(fontsize=8)

        for variant in VARIANT_LABELS:
            pgd = arch_results[variant]["pgd_whitebox"]
            epsilons = [e["epsilon"] * 255 for e in pgd]
            accuracies = [e["accuracy"] for e in pgd]
            line_ax.plot(epsilons, accuracies, marker="o", label=VARIANT_LABELS[variant], color=VARIANT_COLORS[variant])
        line_ax.set_xlabel("epsilon (/255)")
        line_ax.set_ylabel("White-box PGD accuracy")
        line_ax.set_ylim(0, 1.0)
        line_ax.set_title(f"{ARCH_LABELS[arch]}: white-box PGD")
        line_ax.grid(alpha=0.3)
        line_ax.legend(fontsize=8)

    fig.suptitle("QAT training-epochs confound control: does QAT beat equally fine-tuned FP32?")
    fig.tight_layout()
    out_path = REPORTS_DIR / "qat_confound_control.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
