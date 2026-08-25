"""Black-box cross-architecture transfer vs. white-box and FP32->INT8 transfer, PGD only."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "reports"

PAIR_LABELS = {
    "baseline_cnn_to_mobilenet_transfer": "Baseline CNN -> MobileNetV2 (black-box)",
    "mobilenet_transfer_to_baseline_cnn": "MobileNetV2 -> Baseline CNN (black-box)",
}
TARGET_ARCH = {
    "baseline_cnn_to_mobilenet_transfer": "mobilenet_transfer",
    "mobilenet_transfer_to_baseline_cnn": "baseline_cnn",
}
COLORS = {
    "baseline_cnn_to_mobilenet_transfer": "#55A868",
    "mobilenet_transfer_to_baseline_cnn": "#C44E52",
}


def main():
    blackbox = json.loads((REPORTS_DIR / "blackbox_transfer_results.json").read_text())
    white_box = json.loads((REPORTS_DIR / "adversarial_results.json").read_text())

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for pair_key, pair_results in blackbox.items():
        epsilons = [m["epsilon"] * 255 for m in pair_results["pgd"]]
        rates = [m["transfer_attack_success_rate"] for m in pair_results["pgd"]]
        ax.plot(epsilons, rates, marker="o", label=PAIR_LABELS[pair_key], color=COLORS[pair_key])

    for arch_name in ["baseline_cnn", "mobilenet_transfer"]:
        epsilons = [m["epsilon"] * 255 for m in white_box[arch_name]["pgd"]]
        rates = [m["attack_success_rate"] for m in white_box[arch_name]["pgd"]]
        ax.plot(
            epsilons, rates, marker="x", linestyle="--", alpha=0.6,
            label=f"{arch_name} white-box (reference)",
            color="#4C72B0" if arch_name == "baseline_cnn" else "#DD8452",
        )

    ax.set_title("PGD attack success rate: black-box cross-arch transfer vs. white-box")
    ax.set_xlabel("Epsilon (/255, L-inf)")
    ax.set_ylabel("Attack success rate")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_path = REPORTS_DIR / "blackbox_transfer.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
