"""Resolves the QAT training-epochs confound.

ROBUSTNESS_REPORT.md's Phase 4 caveat: QAT's measured robustness gains are confounded
with the extra fine-tuning epochs it receives that PTQ never does. This script isolates
that confound by comparing three variants, all starting from the same FP32 seed-42
checkpoint:

  - fp32_original: the untouched FP32 checkpoint (0 extra epochs).
  - fp32_extra_training: fine-tuned for the identical epochs/optimizer/lr/data/schedule/
    seed as QAT (scripts/train_fp32_extra_training.py), but with no fake quantization.
  - qat: quantization-aware trained (scripts/train_qat.py) with the same protocol.

If QAT still improves clean accuracy / corruption robustness / white-box PGD robustness
relative to fp32_extra_training (not just relative to fp32_original), the improvement is
attributable to quantization-aware training itself, not to the extra epochs.

White-box PGD runs against the differentiable form of each variant (plain FP32 for the
first two, QAT's *prepared* fake-quant surrogate for the third) on CPU throughout, so
device is not a confound either. Corruption sweep is capped to severity 4 (the severity
where PTQ's MobileNetV2 degradation is clearest) to keep this a same-day comparison
rather than the full 4-severity sweep every other script here runs.
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import GTSRBDataset
from src.data.ingest import load_test_dataframe
from src.data.transforms import get_transform
from src.models.evaluate import predict, summarize
from src.models.qat import load_converted, load_prepared
from src.models.registry import archs_with_ckpt
from src.models.wrapper import NormalizedModel
from src.robustness.adversarial import pgd_attack
from src.robustness.corruptions import CORRUPTIONS

RAW_DIR = ROOT / "data" / "raw"
CKPT_DIR = ROOT / "checkpoints"
REPORTS_DIR = ROOT / "reports"

DEVICE = torch.device("cpu")
EPSILONS = [1 / 255, 2 / 255, 4 / 255, 8 / 255]
PGD_STEPS = 10
SEVERITY = 4


def clean_accuracy(model, test_df, mean, std) -> float:
    ds = GTSRBDataset(test_df, transform=get_transform(mean=mean, std=std))
    preds, labels = predict(model, ds, DEVICE, num_workers=0)
    return summarize(preds, labels)["accuracy"]


def corruption_severity4(model, test_df, mean, std) -> dict:
    per_type = {}
    for corr_name, factory in CORRUPTIONS.items():
        transform = get_transform(mean=mean, std=std, corruption_fn=factory(SEVERITY))
        ds = GTSRBDataset(test_df, transform=transform)
        preds, labels = predict(model, ds, DEVICE, num_workers=0)
        per_type[corr_name] = summarize(preds, labels)["accuracy"]
    aggregate = sum(per_type.values()) / len(per_type)
    return {"mean_accuracy": aggregate, "per_type": per_type}


def whitebox_pgd(pixel_space_model, loader) -> list:
    results = []
    for epsilon in EPSILONS:
        total, correct = 0, 0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            adv_images = pgd_attack(pixel_space_model, images, labels, epsilon, alpha=epsilon / 4, steps=PGD_STEPS)
            with torch.no_grad():
                preds = pixel_space_model(adv_images).argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        results.append({"epsilon": epsilon, "accuracy": correct / total})
    return results


def evaluate_variant(model_normalized, pixel_space_model, test_df, mean, std, pgd_loader) -> dict:
    return {
        "clean_accuracy": clean_accuracy(model_normalized, test_df, mean, std),
        "corruption_severity4": corruption_severity4(model_normalized, test_df, mean, std),
        "pgd_whitebox": whitebox_pgd(pixel_space_model, pgd_loader),
    }


def main():
    test_df = load_test_dataframe(RAW_DIR)
    pixel_transform = get_transform(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    pgd_loader = DataLoader(GTSRBDataset(test_df, transform=pixel_transform), batch_size=64, shuffle=False, num_workers=0)

    results = {}
    for arch_name, cfg in archs_with_ckpt().items():
        print(f"\n=== {arch_name} ===")

        fp32_original = cfg["model_fn"]()
        fp32_original.load_state_dict(torch.load(CKPT_DIR / cfg["ckpt"], map_location="cpu"))
        fp32_original = fp32_original.to(DEVICE).eval()
        fp32_original_pixel = NormalizedModel(fp32_original, cfg["mean"], cfg["std"]).eval()

        fp32_extra = cfg["model_fn"]()
        fp32_extra.load_state_dict(
            torch.load(CKPT_DIR / f"{arch_name}_fp32_extra_seed42.pt", map_location="cpu")
        )
        fp32_extra = fp32_extra.to(DEVICE).eval()
        fp32_extra_pixel = NormalizedModel(fp32_extra, cfg["mean"], cfg["std"]).eval()

        qat_converted = load_converted(arch_name, CKPT_DIR / f"{arch_name}_qat_seed42_converted.pt").eval()
        qat_prepared = load_prepared(arch_name, CKPT_DIR / f"{arch_name}_qat_seed42_prepared.pt")
        qat_prepared_pixel = NormalizedModel(qat_prepared, cfg["mean"], cfg["std"]).eval()

        arch_results = {
            "fp32_original": evaluate_variant(
                fp32_original, fp32_original_pixel, test_df, cfg["mean"], cfg["std"], pgd_loader
            ),
            "fp32_extra_training": evaluate_variant(
                fp32_extra, fp32_extra_pixel, test_df, cfg["mean"], cfg["std"], pgd_loader
            ),
            "qat": evaluate_variant(qat_converted, qat_prepared_pixel, test_df, cfg["mean"], cfg["std"], pgd_loader),
        }
        for variant_name, variant_results in arch_results.items():
            print(
                f"  {variant_name}: clean={variant_results['clean_accuracy']:.4f} "
                f"corr_sev4_mean={variant_results['corruption_severity4']['mean_accuracy']:.4f}"
            )
        results[arch_name] = arch_results

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "qat_confound_control.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
