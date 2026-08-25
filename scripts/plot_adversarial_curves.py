"""Accuracy-vs-attack-strength curves — required robustness-report deliverable."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "reports"

ARCH_LABELS = {"baseline_cnn": "Baseline CNN", "mobilenet_transfer": "MobileNetV2 (transfer)"}
ARCH_COLORS = {"baseline_cnn": "#4C72B0", "mobilenet_transfer": "#DD8452"}


def main():
    results = json.loads((REPORTS_DIR / "adversarial_results.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, attack in zip(axes, ["fgsm", "pgd"]):
        for arch_name, arch_results in results.items():
            epsilons = [0.0] + [m["epsilon"] * 255 for m in arch_results[attack]]
            accuracies = [arch_results["clean_accuracy"]] + [m["accuracy"] for m in arch_results[attack]]
            ax.plot(
                epsilons,
                accuracies,
                marker="o",
                label=ARCH_LABELS[arch_name],
                color=ARCH_COLORS[arch_name],
            )
        ax.set_title(attack.upper())
        ax.set_xlabel("Epsilon (/255, L-inf, 0 = clean)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Accuracy vs. adversarial attack strength (full 12,630-image test set)")
    fig.tight_layout()
    out_path = REPORTS_DIR / "adversarial_curves.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")

    # Attack success rate table, printed for the eventual written report
    print("\nAttack success rate (fraction of originally-correct predictions flipped):")
    for arch_name, arch_results in results.items():
        for attack in ["fgsm", "pgd"]:
            rates = [f"{m['epsilon']*255:.0f}/255: {m['attack_success_rate']:.1%}" for m in arch_results[attack]]
            print(f"  {arch_name} {attack}: " + ", ".join(rates))


if __name__ == "__main__":
    main()
