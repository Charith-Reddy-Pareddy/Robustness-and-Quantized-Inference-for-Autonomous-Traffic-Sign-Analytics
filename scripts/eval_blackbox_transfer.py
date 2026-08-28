"""Black-box cross-architecture adversarial transfer.

Threat model gap noted in reports/ROBUSTNESS_REPORT.md: the FP32->INT8 transfer results
attack a model with adversarial examples crafted against a *nearly identical* model (same
architecture, same weights, just quantized) sharing almost the same decision boundary.
This script instead crafts adversarial examples against one FP32 architecture and
evaluates them against the *other*, independently-trained FP32 architecture -- a genuine
black-box transfer with no shared weights or structure between source and target.
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
from src.models.registry import archs_with_ckpt
from src.models.train import get_device, set_seed
from src.models.wrapper import NormalizedModel
from src.robustness.adversarial import fgsm_attack, pgd_attack

RAW_DIR = ROOT / "data" / "raw"
CKPT_DIR = ROOT / "checkpoints"
REPORTS_DIR = ROOT / "reports"

EPSILONS = [1 / 255, 2 / 255, 4 / 255, 8 / 255]
PGD_STEPS = 10

ARCHS = archs_with_ckpt()


def load_model(arch_name, device):
    cfg = ARCHS[arch_name]
    base_model = cfg["model_fn"]()
    base_model.load_state_dict(torch.load(CKPT_DIR / cfg["ckpt"], map_location=device))
    return NormalizedModel(base_model, cfg["mean"], cfg["std"]).to(device).eval()


def run_transfer_eval(source_model, target_model, loader, device, attack: str, epsilon: float):
    total, correct_adv, clean_correct, successful_attacks = 0, 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            clean_correct_mask = target_model(images).argmax(dim=1) == labels

        if attack == "fgsm":
            adv_images = fgsm_attack(source_model, images, labels, epsilon)
        else:
            adv_images = pgd_attack(source_model, images, labels, epsilon, alpha=epsilon / 4, steps=PGD_STEPS)

        with torch.no_grad():
            target_adv_preds = target_model(adv_images).argmax(dim=1)

        total += labels.size(0)
        correct_adv += (target_adv_preds == labels).sum().item()
        clean_correct += clean_correct_mask.sum().item()
        successful_attacks += ((target_adv_preds != labels) & clean_correct_mask).sum().item()

    return {
        "epsilon": epsilon,
        "accuracy": correct_adv / total,
        "transfer_attack_success_rate": successful_attacks / clean_correct if clean_correct else 0.0,
        "clean_correct_count": clean_correct,
    }


def main():
    device = get_device()
    set_seed(42)
    print(f"Using device: {device}")

    test_df = load_test_dataframe(RAW_DIR)
    print(f"Black-box transfer eval set: {len(test_df)} images")

    pixel_transform = get_transform(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ds = GTSRBDataset(test_df, transform=pixel_transform)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    models = {name: load_model(name, device) for name in ARCHS}

    results = {}
    for source_name in ARCHS:
        for target_name in ARCHS:
            if source_name == target_name:
                continue
            pair_key = f"{source_name}_to_{target_name}"
            source_model, target_model = models[source_name], models[target_name]

            with torch.no_grad():
                clean_correct = sum(
                    (target_model(x.to(device)).argmax(1) == y.to(device)).sum().item() for x, y in loader
                )
            clean_accuracy = clean_correct / len(test_df)
            print(f"{pair_key}: target clean accuracy {clean_accuracy:.4f}")

            pair_results = {"target_clean_accuracy": clean_accuracy, "fgsm": [], "pgd": []}
            for epsilon in EPSILONS:
                for attack in ["fgsm", "pgd"]:
                    metrics = run_transfer_eval(source_model, target_model, loader, device, attack, epsilon)
                    pair_results[attack].append(metrics)
                    print(f"{pair_key} {attack} eps={epsilon:.4f}: {metrics}")

            results[pair_key] = pair_results

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "blackbox_transfer_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
