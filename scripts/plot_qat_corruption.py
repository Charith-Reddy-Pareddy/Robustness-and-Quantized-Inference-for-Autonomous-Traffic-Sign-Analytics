"""Severity-4 corruption accuracy across all three quantization variants (FP32, PTQ
static INT8, QAT INT8), all 10 corruption types, both architectures.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.robustness.corruptions import CORRUPTION_CATEGORIES

REPORTS_DIR = ROOT / "reports"
ARCH_LABELS = {"baseline_cnn": "Baseline CNN", "mobilenet_transfer": "MobileNetV2 (transfer)"}
VARIANT_COLORS = {"FP32": "#4C72B0", "PTQ INT8": "#DD8452", "QAT INT8": "#55A868"}
CORRUPTION_ORDER = [name for names in CORRUPTION_CATEGORIES.values() for name in names]


def main():
    fp32 = json.loads((REPORTS_DIR / "corruption_results.json").read_text())
    ptq = json.loads((REPORTS_DIR / "corruption_results_int8.json").read_text())
    qat = json.loads((REPORTS_DIR / "corruption_results_qat.json").read_text())

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    x = np.arange(len(CORRUPTION_ORDER))
    width = 0.26

    for ax, arch in zip(axes, ARCH_LABELS):
        variants = {
            "FP32": [next(e["accuracy"] for e in fp32[arch][c] if e["severity"] == 4) for c in CORRUPTION_ORDER],
            "PTQ INT8": [next(e["accuracy"] for e in ptq[arch][c] if e["severity"] == 4) for c in CORRUPTION_ORDER],
            "QAT INT8": [next(e["accuracy"] for e in qat[arch][c] if e["severity"] == 4) for c in CORRUPTION_ORDER],
        }
        for i, (name, values) in enumerate(variants.items()):
            ax.bar(x + (i - 1) * width, values, width, label=name, color=VARIANT_COLORS[name])
        ax.set_title(ARCH_LABELS[arch])
        ax.set_ylabel("Accuracy at severity 4")
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=9)

    plt.xticks(x, CORRUPTION_ORDER, rotation=30, ha="right")
    fig.suptitle("Severity-4 corruption accuracy: FP32 vs. PTQ-INT8 vs. QAT-INT8")
    fig.tight_layout()
    out_path = REPORTS_DIR / "qat_corruption_comparison.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
