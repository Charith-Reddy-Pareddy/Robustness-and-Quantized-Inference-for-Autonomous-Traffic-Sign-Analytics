"""FP32 vs INT8: corruption robustness and adversarial (transfer) attack success, per
architecture. This is the direct visual for the core hypothesis test."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "reports"

ARCH_LABELS = {"baseline_cnn": "Baseline CNN", "mobilenet_transfer": "MobileNetV2 (transfer)"}


def main():
    corr_fp32 = json.loads((REPORTS_DIR / "corruption_results.json").read_text())
    corr_int8 = json.loads((REPORTS_DIR / "corruption_results_int8.json").read_text())
    adv_fp32 = json.loads((REPORTS_DIR / "adversarial_results.json").read_text())
    adv_int8 = json.loads((REPORTS_DIR / "transfer_attack_results.json").read_text())

    archs = ["baseline_cnn", "mobilenet_transfer"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for row, arch in enumerate(archs):
        ax = axes[row, 0]
        # average accuracy across the 4 corruption types, at each severity, FP32 vs INT8
        for label, data, style in [("FP32", corr_fp32, "-o"), ("INT8", corr_int8, "--s")]:
            severities = [0, 1, 2, 3, 4]
            accs = [data[arch]["clean"]["accuracy"]]
            for sev_idx in range(4):
                vals = [data[arch][c][sev_idx]["accuracy"] for c in ["blur", "noise", "rotation", "brightness_contrast"]]
                accs.append(sum(vals) / len(vals))
            ax.plot(severities, accs, style, label=label)
        ax.set_title(f"{ARCH_LABELS[arch]}: mean corruption accuracy")
        ax.set_xlabel("Severity (0 = clean)")
        ax.set_ylabel("Accuracy (avg over 4 corruptions)")
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend()

        ax = axes[row, 1]
        for label, data, key, style in [
            ("FP32 white-box PGD", adv_fp32, "attack_success_rate", "-o"),
            ("INT8 transfer PGD", adv_int8, "transfer_attack_success_rate", "--s"),
        ]:
            eps = [m["epsilon"] * 255 for m in data[arch]["pgd"]]
            rates = [m[key] for m in data[arch]["pgd"]]
            ax.plot(eps, rates, style, label=label)
        ax.set_title(f"{ARCH_LABELS[arch]}: PGD attack success rate")
        ax.set_xlabel("Epsilon (/255, L-inf)")
        ax.set_ylabel("Attack success rate")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Core hypothesis test: does INT8 quantization change robustness?")
    fig.tight_layout()
    out_path = REPORTS_DIR / "quantization_comparison.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
