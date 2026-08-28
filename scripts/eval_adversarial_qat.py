"""Two adversarial evaluations against the QAT models, seed 42, full test set:

1. True INT8 white-box: FGSM/PGD computed directly against the *prepared* (fake-quant,
   still-differentiable) QAT model's own gradients. This is the threat model
   ROBUSTNESS_REPORT.md's Threat Models section identifies as untested for PTQ -- the
   fake-quant straight-through estimator makes this a genuine white-box attack on a
   quantization-aware model, not a transfer attack.
2. FP32->QAT-INT8 transfer: the same FP32-crafted adversarial examples used elsewhere in
   this project, evaluated unchanged against the *converted* (real INT8) QAT model --
   directly comparable to eval_transfer_attacks.py's FP32->PTQ-INT8 numbers.
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
from src.models.qat import load_converted, load_prepared
from src.models.registry import ARCH_SPECS
from src.models.train import get_device, set_seed
from src.models.wrapper import NormalizedModel
from src.robustness.adversarial import fgsm_attack, pgd_attack

RAW_DIR = ROOT / "data" / "raw"
CKPT_DIR = ROOT / "checkpoints"
REPORTS_DIR = ROOT / "reports"

EPSILONS = [1 / 255, 2 / 255, 4 / 255, 8 / 255]
PGD_STEPS = 10

ARCHS = {
    name: {"fp32_fn": spec["model_fn"], "mean": spec["mean"], "std": spec["std"]}
    for name, spec in ARCH_SPECS.items()
}


def run_attack(model, loader, device, attack: str, epsilon: float, target_model=None):
    """Attack `model`; if target_model is given, evaluate the resulting perturbation
    against target_model instead (transfer), using target_model's own clean predictions
    as the denominator -- else evaluate white-box against `model` itself."""
    eval_model = target_model if target_model is not None else model
    total, correct_adv, clean_correct, successful_attacks = 0, 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        eval_images = images.cpu() if target_model is not None else images
        eval_labels = labels.cpu() if target_model is not None else labels
        with torch.no_grad():
            clean_correct_mask = eval_model(eval_images).argmax(dim=1) == eval_labels

        if attack == "fgsm":
            adv_images = fgsm_attack(model, images, labels, epsilon)
        else:
            adv_images = pgd_attack(model, images, labels, epsilon, alpha=epsilon / 4, steps=PGD_STEPS)

        adv_eval = adv_images.cpu() if target_model is not None else adv_images
        with torch.no_grad():
            adv_preds = eval_model(adv_eval).argmax(dim=1)

        total += labels.size(0)
        correct_adv += (adv_preds == eval_labels).sum().item()
        clean_correct += clean_correct_mask.sum().item()
        successful_attacks += ((adv_preds != eval_labels) & clean_correct_mask).sum().item()

    return {
        "epsilon": epsilon,
        "accuracy": correct_adv / total,
        "attack_success_rate": successful_attacks / clean_correct if clean_correct else 0.0,
        "clean_correct_count": clean_correct,
    }


def main():
    device = get_device()
    set_seed(42)
    print(f"Attack-generation device: {device}; QAT converted-model eval device: cpu")

    test_df = load_test_dataframe(RAW_DIR)
    pixel_transform = get_transform(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ds = GTSRBDataset(test_df, transform=pixel_transform)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    white_box_results = {}
    transfer_results = {}
    for arch_name, cfg in ARCHS.items():
        print(f"\n=== {arch_name} ===")

        # 1. True INT8 white-box, on the prepared (differentiable) QAT model. QAT's
        # fake-quant observer op isn't implemented for MPS, so this runs entirely on CPU
        # -- unlike the FP32 attack-generation device used below for the transfer case.
        prepared = load_prepared(arch_name, CKPT_DIR / f"{arch_name}_qat_seed42_prepared.pt")
        prepared_wrapped = NormalizedModel(prepared, cfg["mean"], cfg["std"]).eval()
        arch_wb = {"fgsm": [], "pgd": []}
        for epsilon in EPSILONS:
            for attack in ["fgsm", "pgd"]:
                metrics = run_attack(prepared_wrapped, loader, torch.device("cpu"), attack, epsilon)
                arch_wb[attack].append(metrics)
                print(f"{arch_name} QAT white-box {attack} eps={epsilon:.4f}: {metrics}")
        white_box_results[arch_name] = arch_wb

        # 2. FP32-crafted examples transferred to the converted (real INT8) QAT model.
        fp32_model = cfg["fp32_fn"]()
        fp32_model.load_state_dict(torch.load(CKPT_DIR / f"{arch_name}_seed42.pt", map_location=device))
        fp32_wrapped = NormalizedModel(fp32_model, cfg["mean"], cfg["std"]).to(device).eval()

        converted = load_converted(arch_name, CKPT_DIR / f"{arch_name}_qat_seed42_converted.pt")
        converted_wrapped = NormalizedModel(converted, cfg["mean"], cfg["std"]).eval()

        arch_tr = {"fgsm": [], "pgd": []}
        for epsilon in EPSILONS:
            for attack in ["fgsm", "pgd"]:
                metrics = run_attack(
                    fp32_wrapped, loader, device, attack, epsilon, target_model=converted_wrapped
                )
                arch_tr[attack].append(metrics)
                print(f"{arch_name} FP32->QAT transfer {attack} eps={epsilon:.4f}: {metrics}")
        transfer_results[arch_name] = arch_tr

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "adversarial_results_qat_whitebox.json").write_text(json.dumps(white_box_results, indent=2))
    (REPORTS_DIR / "adversarial_results_qat_transfer.json").write_text(json.dumps(transfer_results, indent=2))
    print("Saved QAT adversarial results")


if __name__ == "__main__":
    main()
