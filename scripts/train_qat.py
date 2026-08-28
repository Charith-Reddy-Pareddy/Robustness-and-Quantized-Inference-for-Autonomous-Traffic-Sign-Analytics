"""Quantization-aware training (QAT), fine-tuned from the existing FP32 seed-42
checkpoints. Third quantization variant alongside FP32 and PTQ static INT8 (ONNX Runtime).

QAT fake-quant ops (_fused_moving_avg_obs_fq_helper) aren't implemented for MPS, so this
runs on CPU throughout -- fine given fine-tuning is only a few epochs from an
already-converged FP32 starting point, unlike the original 12-epoch from-scratch training.

Saves two artifacts per architecture (whole pickled modules, not just state_dicts, since
reconstructing a converted-quantized module's exact structure independently is fiddly):
  - {arch}_qat_seed42_prepared.pt: fake-quantized but still FP32-tensor and differentiable
    -- this is the "quantization-aware differentiable surrogate" the Threat Models section
    of ROBUSTNESS_REPORT.md identifies as needed for a true INT8 white-box attack.
  - {arch}_qat_seed42_converted.pt: real INT8 (qnnpack backend), for clean/corruption/
    latency evaluation, comparable to the PTQ INT8 ONNX models elsewhere in this project.
"""

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import GTSRBDataset
from src.data.ingest import load_test_dataframe, load_train_dataframe, track_aware_split
from src.data.transforms import get_transform
from src.models.evaluate import predict, summarize
from src.models.qat import BUILD_FNS, convert_to_quantized, prepare_for_qat
from src.models.registry import archs_with_ckpt

RAW_DIR = ROOT / "data" / "raw"
CKPT_DIR = ROOT / "checkpoints"

QAT_EPOCHS = 3
QAT_LR = 1e-4
BATCH_SIZE = 128

ARCHS = {
    name: {"build_fn": BUILD_FNS[name], "fp32_ckpt": spec["ckpt"], "mean": spec["mean"], "std": spec["std"]}
    for name, spec in archs_with_ckpt().items()
}


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(mode=training)
    criterion = nn.CrossEntropyLoss()
    total_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        all_preds.append(logits.argmax(dim=1).detach().cpu())
        all_labels.append(labels.detach().cpu())
    preds = torch.cat(all_preds).numpy()
    labels_arr = torch.cat(all_labels).numpy()
    return total_loss / len(loader.dataset), f1_score(labels_arr, preds, average="macro")


def main():
    device = torch.device("cpu")
    print(f"QAT fine-tuning device: {device} (MPS doesn't support fake-quant observer ops)")

    train_df = load_train_dataframe(RAW_DIR)
    train_split_df, val_split_df = track_aware_split(train_df, val_fraction=0.15, seed=42)
    test_df = load_test_dataframe(RAW_DIR)

    for arch_name, cfg in ARCHS.items():
        converted_path = CKPT_DIR / f"{arch_name}_qat_seed42_converted.pt"
        if converted_path.exists():
            print(f"\n=== {arch_name}: already done, skipping ===")
            continue

        print(f"\n=== {arch_name} ===")
        transform = get_transform(mean=cfg["mean"], std=cfg["std"])
        train_ds = GTSRBDataset(train_split_df, transform=transform)
        val_ds = GTSRBDataset(val_split_df, transform=transform)
        test_ds = GTSRBDataset(test_df, transform=transform)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = cfg["build_fn"](num_classes=43)
        model.load_state_dict(torch.load(CKPT_DIR / cfg["fp32_ckpt"], map_location="cpu"))
        model = prepare_for_qat(model).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=QAT_LR)
        best_val_f1, best_state = -1.0, None
        for epoch in range(QAT_EPOCHS):
            train_loss, train_f1 = run_epoch(model, train_loader, device, optimizer)
            val_loss, val_f1 = run_epoch(model, val_loader, device, optimizer=None)
            print(
                f"epoch {epoch}: train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
                f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
            )
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)

        prepared_path = CKPT_DIR / f"{arch_name}_qat_seed42_prepared.pt"
        torch.save(model.state_dict(), prepared_path)
        print(f"Saved prepared (differentiable, fake-quant) state_dict to {prepared_path}")

        converted = convert_to_quantized(model)
        converted_path = CKPT_DIR / f"{arch_name}_qat_seed42_converted.pt"
        torch.save(converted.state_dict(), converted_path)
        print(f"Saved converted (real INT8) state_dict to {converted_path}")

        preds, labels = predict(converted, test_ds, torch.device("cpu"), num_workers=0)
        results = summarize(preds, labels)
        print(f"{arch_name} QAT INT8 test accuracy: {results['accuracy']:.4f}, macro-F1: {results['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
