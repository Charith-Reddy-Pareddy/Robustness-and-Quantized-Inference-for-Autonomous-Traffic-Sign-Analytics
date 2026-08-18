"""FGSM/PGD adversarial evaluation on the FP32 models (seed 42), across epsilon strengths.

Uses a stratified subsample of the Test set (not the full 12,630 images): PGD requires a
forward+backward pass per step, and at 10 steps x 4 epsilons x 2 models the full test set
would take a long time for limited extra statistical value. 2000 stratified samples keeps
per-class representation while making iterative attacks tractable.
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
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, NORM_MEAN, NORM_STD, get_transform
from src.models.baseline_cnn import BaselineCNN
from src.models.train import get_device, set_seed
from src.models.transfer_model import build_mobilenet
from src.models.wrapper import NormalizedModel
from src.robustness.adversarial import fgsm_attack, pgd_attack

RAW_DIR = ROOT / "data" / "raw"
CKPT_DIR = ROOT / "checkpoints"
REPORTS_DIR = ROOT / "reports"

N_SUBSAMPLE = 2000
EPSILONS = [1 / 255, 2 / 255, 4 / 255, 8 / 255]
PGD_STEPS = 10

ARCHS = {
    "baseline_cnn": {
        "model_fn": lambda: BaselineCNN(num_classes=43),
        "mean": NORM_MEAN,
        "std": NORM_STD,
        "ckpt": "baseline_cnn_seed42.pt",
    },
    "mobilenet_transfer": {
        "model_fn": lambda: build_mobilenet(num_classes=43),
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "ckpt": "mobilenet_transfer_seed42.pt",
    },
}


def stratified_subsample(df, n_total: int, seed: int):
    frac_df = df.groupby("ClassId", group_keys=False).apply(
        lambda g: g.sample(max(1, round(len(g) * n_total / len(df))), random_state=seed)
    )
    return frac_df.reset_index(drop=True)


def run_attack_eval(model, loader, device, attack: str, epsilon: float):
    total, correct_adv, clean_correct, successful_attacks = 0, 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            clean_correct_mask = model(images).argmax(dim=1) == labels

        if attack == "fgsm":
            adv_images = fgsm_attack(model, images, labels, epsilon)
        else:
            adv_images = pgd_attack(model, images, labels, epsilon, alpha=epsilon / 4, steps=PGD_STEPS)

        with torch.no_grad():
            adv_preds = model(adv_images).argmax(dim=1)

        total += labels.size(0)
        correct_adv += (adv_preds == labels).sum().item()
        clean_correct += clean_correct_mask.sum().item()
        successful_attacks += ((adv_preds != labels) & clean_correct_mask).sum().item()

    return {
        "epsilon": epsilon,
        "accuracy": correct_adv / total,
        "attack_success_rate": successful_attacks / clean_correct if clean_correct else 0.0,
        "clean_correct_count": clean_correct,
    }


def main():
    device = get_device()
    set_seed(42)
    print(f"Using device: {device}")

    test_df = load_test_dataframe(RAW_DIR)
    sub_df = stratified_subsample(test_df, N_SUBSAMPLE, seed=42)
    print(f"Adversarial eval subsample: {len(sub_df)} images across {sub_df['ClassId'].nunique()} classes")

    results = {}
    for arch_name, cfg in ARCHS.items():
        base_model = cfg["model_fn"]()
        base_model.load_state_dict(torch.load(CKPT_DIR / cfg["ckpt"], map_location=device))
        model = NormalizedModel(base_model, cfg["mean"], cfg["std"]).to(device).eval()

        pixel_transform = get_transform(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        ds = GTSRBDataset(sub_df, transform=pixel_transform)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

        with torch.no_grad():
            clean_correct = sum(
                (model(x.to(device)).argmax(1) == y.to(device)).sum().item() for x, y in loader
            )
        clean_accuracy = clean_correct / len(sub_df)
        print(f"{arch_name} clean subsample accuracy: {clean_accuracy:.4f}")

        arch_results = {"clean_accuracy": clean_accuracy, "fgsm": [], "pgd": []}
        for epsilon in EPSILONS:
            for attack in ["fgsm", "pgd"]:
                metrics = run_attack_eval(model, loader, device, attack, epsilon)
                arch_results[attack].append(metrics)
                print(f"{arch_name} {attack} eps={epsilon:.4f}: {metrics}")

        results[arch_name] = arch_results

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "adversarial_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
