"""Statistical rigor pass: mean/std/CI per condition, and bootstrap CI + paired
significance test + effect size for every FP32-vs-INT8 robustness delta, across the
5-seed corruption and PGD-adversarial multi-seed results.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.stats import bootstrap_paired_delta_ci, mean_std_ci

REPORTS_DIR = ROOT / "reports"
SEEDS = [42, 123, 2024, 7, 999]
CORRUPTIONS = ["blur", "noise", "rotation", "brightness_contrast"]
SEVERITIES = [1, 2, 3, 4]
EPSILONS = [1 / 255, 2 / 255, 4 / 255, 8 / 255]
ARCHS = ["baseline_cnn", "mobilenet_transfer"]


def corruption_accuracies(results, arch, corr_name, severity):
    return [
        next(m["accuracy"] for m in results[arch][str(seed)][corr_name] if m["severity"] == severity)
        for seed in SEEDS
    ]


def adversarial_rates(results, arch, epsilon, rate_key):
    return [
        next(m[rate_key] for m in results[arch][str(seed)] if abs(m["epsilon"] - epsilon) < 1e-9)
        for seed in SEEDS
    ]


def main():
    fp32_corruption = json.loads((REPORTS_DIR / "corruption_results_multiseed_fp32.json").read_text())
    int8_corruption = json.loads((REPORTS_DIR / "corruption_results_multiseed_int8.json").read_text())
    white_box = json.loads((REPORTS_DIR / "adversarial_results_multiseed_whitebox.json").read_text())
    transfer = json.loads((REPORTS_DIR / "adversarial_results_multiseed_transfer.json").read_text())

    output = {"corruption": {}, "adversarial_pgd": {}}

    for arch in ARCHS:
        output["corruption"][arch] = {}
        for corr_name in CORRUPTIONS:
            output["corruption"][arch][corr_name] = {}
            for severity in SEVERITIES:
                fp32_vals = corruption_accuracies(fp32_corruption, arch, corr_name, severity)
                int8_vals = corruption_accuracies(int8_corruption, arch, corr_name, severity)
                output["corruption"][arch][corr_name][str(severity)] = {
                    "fp32": mean_std_ci(fp32_vals),
                    "int8": mean_std_ci(int8_vals),
                    "delta": bootstrap_paired_delta_ci(fp32_vals, int8_vals),
                }

        output["adversarial_pgd"][arch] = {}
        for epsilon in EPSILONS:
            wb_vals = adversarial_rates(white_box, arch, epsilon, "attack_success_rate")
            tr_vals = adversarial_rates(transfer, arch, epsilon, "transfer_attack_success_rate")
            output["adversarial_pgd"][arch][f"{epsilon:.4f}"] = {
                "white_box": mean_std_ci(wb_vals),
                "int8_transfer": mean_std_ci(tr_vals),
                "delta": bootstrap_paired_delta_ci(wb_vals, tr_vals),
            }

    out_path = REPORTS_DIR / "statistics.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved {out_path}")

    # Highlight: the reviewer's literal example
    example = output["corruption"]["mobilenet_transfer"]["brightness_contrast"]["4"]
    print("\nMobileNetV2 brightness/contrast severity 4:")
    print(f"  FP32: {example['fp32']['mean']*100:.2f}% (95% CI [{example['fp32']['ci_low']*100:.2f}, {example['fp32']['ci_high']*100:.2f}])")
    print(f"  INT8: {example['int8']['mean']*100:.2f}% (95% CI [{example['int8']['ci_low']*100:.2f}, {example['int8']['ci_high']*100:.2f}])")
    print(f"  Delta: {example['delta']['mean_delta']*100:.2f}pp, bootstrap 95% CI [{example['delta']['bootstrap_ci_low']*100:.2f}, {example['delta']['bootstrap_ci_high']*100:.2f}]")
    print(f"  Paired t-test p={example['delta']['paired_t_pvalue']:.4f}, Cohen's dz={example['delta']['cohens_dz']:.2f}")


if __name__ == "__main__":
    main()
