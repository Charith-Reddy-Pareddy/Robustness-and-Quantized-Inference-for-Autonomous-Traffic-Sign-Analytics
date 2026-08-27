"""PGD attack success across all four adversarial threat models now covered:
FP32 white-box, FP32->PTQ-INT8 transfer, true QAT-INT8 white-box (via the
differentiable fake-quant surrogate), and FP32->QAT-INT8 transfer.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "reports"

ARCH_LABELS = {"baseline_cnn": "Baseline CNN", "mobilenet_transfer": "MobileNetV2 (transfer)"}
SERIES_STYLE = {
    "FP32 white-box": {"color": "#4C72B0", "linestyle": "-", "marker": "o"},
    "PTQ-INT8 transfer": {"color": "#DD8452", "linestyle": "-", "marker": "o"},
    "QAT-INT8 white-box": {"color": "#55A868", "linestyle": "-", "marker": "s"},
    "FP32->QAT-INT8 transfer": {"color": "#55A868", "linestyle": "--", "marker": "^"},
}


def main():
    fp32 = json.loads((REPORTS_DIR / "adversarial_results.json").read_text())
    ptq_transfer = json.loads((REPORTS_DIR / "transfer_attack_results.json").read_text())
    qat_wb = json.loads((REPORTS_DIR / "adversarial_results_qat_whitebox.json").read_text())
    qat_transfer = json.loads((REPORTS_DIR / "adversarial_results_qat_transfer.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, arch in zip(axes, ARCH_LABELS):
        series = {
            "FP32 white-box": [m["attack_success_rate"] for m in fp32[arch]["pgd"]],
            "PTQ-INT8 transfer": [m["transfer_attack_success_rate"] for m in ptq_transfer[arch]["pgd"]],
            "QAT-INT8 white-box": [m["attack_success_rate"] for m in qat_wb[arch]["pgd"]],
            "FP32->QAT-INT8 transfer": [m["attack_success_rate"] for m in qat_transfer[arch]["pgd"]],
        }
        epsilons = [m["epsilon"] * 255 for m in fp32[arch]["pgd"]]
        for name, values in series.items():
            style = SERIES_STYLE[name]
            ax.plot(epsilons, values, label=name, **style)
        ax.set_title(ARCH_LABELS[arch])
        ax.set_xlabel("Epsilon (/255, L-inf)")
        ax.set_ylabel("PGD attack success rate")
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower right")

    fig.suptitle("PGD attack success across all four adversarial threat models")
    fig.tight_layout()
    out_path = REPORTS_DIR / "qat_adversarial_comparison.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
