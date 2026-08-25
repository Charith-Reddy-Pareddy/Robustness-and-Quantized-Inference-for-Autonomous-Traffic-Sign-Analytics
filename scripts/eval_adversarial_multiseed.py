"""PGD adversarial robustness (FP32 white-box + FP32->INT8 transfer) across all 5 seeds.

PGD only, not FGSM: PGD is the stronger, more informative attack in this project's
findings (see ROBUSTNESS_REPORT.md's gradient-masking discussion), and running both
attacks across 5 seeds would roughly double an already-expensive full-test-set sweep for
limited additional statistical value. Documented as a scoping choice in the report.

Requires scripts/quantize_models_multiseed.py to have been run first for the INT8 side.
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
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
from src.robustness.adversarial import pgd_attack

RAW_DIR = ROOT / "data" / "raw"
CKPT_DIR = ROOT / "checkpoints"
ONNX_DIR = ROOT / "onnx" / "multiseed"
REPORTS_DIR = ROOT / "reports"

SEEDS = [42, 123, 2024, 7, 999]
EPSILONS = [1 / 255, 2 / 255, 4 / 255, 8 / 255]
PGD_STEPS = 10

ARCHS = {
    "baseline_cnn": {
        "model_fn": lambda: BaselineCNN(num_classes=43),
        "mean": NORM_MEAN,
        "std": NORM_STD,
    },
    "mobilenet_transfer": {
        "model_fn": lambda: build_mobilenet(num_classes=43),
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
}


def run_white_box(model, loader, device, epsilon):
    total, correct_adv, clean_correct, successful_attacks = 0, 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            clean_correct_mask = model(images).argmax(dim=1) == labels
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
    }


def run_transfer(fp32_model, onnx_sess, input_name, loader, device, epsilon):
    total, correct_adv, clean_correct, successful_attacks = 0, 0, 0, 0
    for images, labels in loader:
        images_gpu, labels_gpu = images.to(device), labels.to(device)
        int8_clean_logits = onnx_sess.run(None, {input_name: images.numpy().astype(np.float32)})[0]
        clean_correct_mask = int8_clean_logits.argmax(axis=1) == labels.numpy()
        adv_images = pgd_attack(fp32_model, images_gpu, labels_gpu, epsilon, alpha=epsilon / 4, steps=PGD_STEPS)
        int8_adv_logits = onnx_sess.run(None, {input_name: adv_images.cpu().numpy().astype(np.float32)})[0]
        int8_adv_preds = int8_adv_logits.argmax(axis=1)
        total += labels.size(0)
        correct_adv += (int8_adv_preds == labels.numpy()).sum()
        clean_correct += clean_correct_mask.sum()
        successful_attacks += ((int8_adv_preds != labels.numpy()) & clean_correct_mask).sum()
    return {
        "epsilon": epsilon,
        "accuracy": float(correct_adv / total),
        "transfer_attack_success_rate": float(successful_attacks / clean_correct) if clean_correct else 0.0,
    }


def main():
    device = get_device()
    set_seed(42)
    print(f"Using device: {device}")

    test_df = load_test_dataframe(RAW_DIR)
    pixel_transform = get_transform(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ds = GTSRBDataset(test_df, transform=pixel_transform)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    white_box_results = {}
    transfer_results = {}
    for arch_name, cfg in ARCHS.items():
        white_box_results[arch_name] = {}
        transfer_results[arch_name] = {}
        for seed in SEEDS:
            base_model = cfg["model_fn"]()
            base_model.load_state_dict(
                torch.load(CKPT_DIR / f"{arch_name}_seed{seed}.pt", map_location=device)
            )
            fp32_model = NormalizedModel(base_model, cfg["mean"], cfg["std"]).to(device).eval()

            white_box_results[arch_name][str(seed)] = [
                run_white_box(fp32_model, loader, device, eps) for eps in EPSILONS
            ]
            print(f"{arch_name} seed{seed} white-box: {white_box_results[arch_name][str(seed)]}")

            onnx_path = ONNX_DIR / f"{arch_name}_seed{seed}_int8.onnx"
            onnx_sess = ort.InferenceSession(str(onnx_path))
            input_name = onnx_sess.get_inputs()[0].name
            transfer_results[arch_name][str(seed)] = [
                run_transfer(fp32_model, onnx_sess, input_name, loader, device, eps) for eps in EPSILONS
            ]
            print(f"{arch_name} seed{seed} transfer: {transfer_results[arch_name][str(seed)]}")

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "adversarial_results_multiseed_whitebox.json").write_text(
        json.dumps(white_box_results, indent=2)
    )
    (REPORTS_DIR / "adversarial_results_multiseed_transfer.json").write_text(
        json.dumps(transfer_results, indent=2)
    )
    print("Saved multi-seed adversarial results")


if __name__ == "__main__":
    main()
