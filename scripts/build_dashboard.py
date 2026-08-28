"""Build the static metrics dashboard (docs/index.html, served via GitHub Pages) from
dashboard/template.html plus the current reports/*.json results.

Re-run this any time the underlying results change (e.g. once the QAT evaluation lands)
and commit the regenerated docs/index.html -- it's a plain static file, not built by CI,
so GitHub Pages just serves whatever was last checked in.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
TEMPLATE_PATH = ROOT / "dashboard" / "template.html"
OUT_PATH = ROOT / "docs" / "index.html"


def load(name):
    return json.loads((REPORTS_DIR / name).read_text())


def build_data():
    ms = load("multi_seed_summary.json")
    corr_fp32 = load("corruption_results.json")
    corr_int8 = load("corruption_results_int8.json")
    adv = load("adversarial_results.json")
    tr = load("transfer_attack_results.json")
    bb = load("blackbox_transfer_results.json")
    mapillary = load("mapillary_generalization_results.json")
    belgium = load("belgium_generalization_results.json")
    latency = load("latency_benchmark.json")
    stats = load("statistics.json")
    corr_qat = load("corruption_results_qat.json")
    qat_wb = load("adversarial_results_qat_whitebox.json")
    qat_tr = load("adversarial_results_qat_transfer.json")
    latency_qat = load("latency_benchmark_qat.json")

    archs = ["baseline_cnn", "mobilenet_transfer"]
    corruptions = ["blur", "noise", "rotation", "brightness_contrast"]

    out = {
        "multi_seed": ms,
        "clean": {
            a: {"fp32": corr_fp32[a]["clean"], "int8": corr_int8[a]["clean"]} for a in archs
        },
        "corruption": {
            a: {
                c: {
                    "fp32": [{"severity": m["severity"], "accuracy": m["accuracy"]} for m in corr_fp32[a][c]],
                    "int8": [{"severity": m["severity"], "accuracy": m["accuracy"]} for m in corr_int8[a][c]],
                }
                for c in corruptions
            }
            for a in archs
        },
        "adversarial": {
            a: {
                "whitebox_pgd": [{"epsilon": m["epsilon"], "success": m["attack_success_rate"]} for m in adv[a]["pgd"]],
                "transfer_pgd": [
                    {"epsilon": m["epsilon"], "success": m["transfer_attack_success_rate"]} for m in tr[a]["pgd"]
                ],
            }
            for a in archs
        },
        "blackbox": {
            k: {"pgd": [{"epsilon": m["epsilon"], "success": m["transfer_attack_success_rate"]} for m in v["pgd"]]}
            for k, v in bb.items()
        },
        "mapillary": {
            a: {
                "gtsrb": mapillary[a]["gtsrb_test_accuracy_restricted"],
                "fp32": mapillary[a]["fp32_accuracy"],
                "int8": mapillary[a]["int8_accuracy"],
            }
            for a in archs
        },
        "belgium": {
            a: {
                "gtsrb": belgium[a]["gtsrb_test_accuracy_restricted"],
                "fp32": belgium[a]["fp32_accuracy"],
                "int8": belgium[a]["int8_accuracy"],
            }
            for a in archs
        },
        "latency": {
            a: {
                "fp32_ms": latency[a]["fp32"]["mean_ms"],
                "int8_ms": latency[a]["int8"]["mean_ms"],
                "fp32_kb": latency[a]["fp32"]["size_kb"],
                "int8_kb": latency[a]["int8"]["size_kb"],
            }
            for a in archs
        },
        "statistics_example": stats["corruption"]["mobilenet_transfer"]["brightness_contrast"]["4"],
        "qat_corruption_sev4": {
            a: {
                c: {
                    "fp32": next(m["accuracy"] for m in corr_fp32[a][c] if m["severity"] == 4),
                    "ptq": next(m["accuracy"] for m in corr_int8[a][c] if m["severity"] == 4),
                    "qat": next(m["accuracy"] for m in corr_qat[a][c] if m["severity"] == 4),
                }
                for c in ["blur", "brightness_contrast"]
            }
            for a in archs
        },
        "qat_adversarial": {
            a: {
                "whitebox_pgd": [m["attack_success_rate"] for m in adv[a]["pgd"]],
                "ptq_transfer_pgd": [m["transfer_attack_success_rate"] for m in tr[a]["pgd"]],
                "qat_whitebox_pgd": [m["attack_success_rate"] for m in qat_wb[a]["pgd"]],
                "qat_transfer_pgd": [m["attack_success_rate"] for m in qat_tr[a]["pgd"]],
            }
            for a in archs
        },
        "qat_clean": {
            a: {"fp32": corr_fp32[a]["clean"]["accuracy"], "ptq": corr_int8[a]["clean"]["accuracy"], "qat": corr_qat[a]["clean"]["accuracy"]}
            for a in archs
        },
        "qat_latency": {
            a: {
                "fp32_ms": latency_qat[a]["fp32"]["mean_ms"],
                "qat_ms": latency_qat[a]["qat_int8"]["mean_ms"],
                "fp32_kb": latency_qat[a]["fp32"]["size_kb"],
                "qat_kb": latency_qat[a]["qat_int8"]["size_kb"],
            }
            for a in archs
        },
    }
    return out


def main():
    data = build_data()
    template = TEMPLATE_PATH.read_text()
    if "__DATA_JSON__" not in template:
        print("ERROR: placeholder __DATA_JSON__ not found in template", file=sys.stderr)
        sys.exit(1)

    body = template.replace("__DATA_JSON__", json.dumps(data))
    # GitHub Pages serves plain static files -- unlike an Artifact publish, there's no
    # automatic doctype/html/body wrapper, so the template's fragment needs one here.
    # Everything through </style> (title, font links, styles) becomes <head> content.
    head_extra_end = body.index("</style>") + len("</style>")
    doc = (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        + body[:head_extra_end]
        + "\n</head>\n<body>\n"
        + body[head_extra_end:]
        + "\n</body>\n</html>\n"
    )

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(doc)
    print(f"Saved {OUT_PATH} ({len(doc)} bytes)")


if __name__ == "__main__":
    main()
