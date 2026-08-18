"""Transfer-attack the INT8 ONNX models with perturbations generated against the FP32
PyTorch models (Phase 2's central comparison).

Methodology note (per spec): FGSM/PGD need a differentiable model, and the quantized ONNX
graph isn't one, so adversarial examples are generated white-box against the FP32 PyTorch
model, then fed unchanged to the INT8 ONNX model. This isn't just a workaround -- it's a
realistic threat model, since an attacker deployed against a quantized model in production
generally won't have white-box access to its internals either.
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
from src.robustness.adversarial import fgsm_attack, pgd_attack

RAW_DIR = ROOT / "data" / "raw"
CKPT_DIR = ROOT / "checkpoints"
ONNX_DIR = ROOT / "onnx"
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


def run_transfer_eval(fp32_model, onnx_sess, input_name, loader, device, attack: str, epsilon: float):
    total, correct_adv, clean_correct, successful_attacks = 0, 0, 0, 0
    for images, labels in loader:
        images_gpu, labels_gpu = images.to(device), labels.to(device)

        int8_clean_logits = onnx_sess.run(None, {input_name: images.numpy().astype(np.float32)})[0]
        clean_correct_mask = int8_clean_logits.argmax(axis=1) == labels.numpy()

        if attack == "fgsm":
            adv_images = fgsm_attack(fp32_model, images_gpu, labels_gpu, epsilon)
        else:
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
        "clean_correct_count": int(clean_correct),
    }


def main():
    device = get_device()
    set_seed(42)
    print(f"Using device: {device}")

    test_df = load_test_dataframe(RAW_DIR)
    sub_df = stratified_subsample(test_df, N_SUBSAMPLE, seed=42)
    print(f"Transfer-attack eval subsample: {len(sub_df)} images")

    results = {}
    for arch_name, cfg in ARCHS.items():
        base_model = cfg["model_fn"]()
        base_model.load_state_dict(torch.load(CKPT_DIR / cfg["ckpt"], map_location=device))
        fp32_model = NormalizedModel(base_model, cfg["mean"], cfg["std"]).to(device).eval()

        onnx_sess = ort.InferenceSession(str(ONNX_DIR / f"{arch_name}_int8.onnx"))
        input_name = onnx_sess.get_inputs()[0].name

        pixel_transform = get_transform(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        ds = GTSRBDataset(sub_df, transform=pixel_transform)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

        clean_correct = 0
        for images, labels in loader:
            logits = onnx_sess.run(None, {input_name: images.numpy().astype(np.float32)})[0]
            clean_correct += (logits.argmax(axis=1) == labels.numpy()).sum()
        clean_accuracy = clean_correct / len(sub_df)
        print(f"{arch_name} INT8 clean subsample accuracy: {clean_accuracy:.4f}")

        arch_results = {"clean_accuracy": float(clean_accuracy), "fgsm": [], "pgd": []}
        for epsilon in EPSILONS:
            for attack in ["fgsm", "pgd"]:
                metrics = run_transfer_eval(fp32_model, onnx_sess, input_name, loader, device, attack, epsilon)
                arch_results[attack].append(metrics)
                print(f"{arch_name} transfer-{attack} eps={epsilon:.4f}: {metrics}")

        results[arch_name] = arch_results

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "transfer_attack_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
