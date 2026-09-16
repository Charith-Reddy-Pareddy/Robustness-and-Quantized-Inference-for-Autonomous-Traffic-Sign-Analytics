"""FP32 equal-training control for the QAT confound.

train_qat.py's QAT numbers are confounded with the extra fine-tuning epochs QAT receives
that PTQ never does -- ROBUSTNESS_REPORT.md's Phase 4 caveat says as much. This script
isolates that confound: it fine-tunes the same FP32 seed-42 checkpoints for the identical
epochs/optimizer/lr/batch-size/data-split/seed as train_qat.py (see
src/models/finetune_config.py, the single source of truth both scripts read from), but
with no fake-quantization at all. The only difference between this script and
train_qat.py is prepare_for_qat/convert_to_quantized -- everything else, including the
per-arch set_seed call, is identical.

Comparing FP32_original vs. this FP32_extra_training vs. QAT (scripts/eval_qat_confound_control.py)
tells you whether QAT's measured robustness gains come from quantization-aware training
itself, or just from three more epochs of fine-tuning.
"""

import copy
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import GTSRBDataset
from src.data.ingest import load_test_dataframe, load_train_dataframe, track_aware_split
from src.data.transforms import get_transform
from src.models.evaluate import predict, summarize
from src.models.finetune_config import FINE_TUNE_BATCH_SIZE, FINE_TUNE_EPOCHS, FINE_TUNE_LR, FINE_TUNE_SEED
from src.models.registry import archs_with_ckpt
from src.models.train import run_epoch, set_seed

RAW_DIR = ROOT / "data" / "raw"
CKPT_DIR = ROOT / "checkpoints"

ARCHS = archs_with_ckpt()


def main():
    device = torch.device("cpu")
    print(f"FP32 extra-training device: {device} (matches train_qat.py's CPU-only execution)")

    train_df = load_train_dataframe(RAW_DIR)
    train_split_df, val_split_df = track_aware_split(train_df, val_fraction=0.15, seed=42)
    test_df = load_test_dataframe(RAW_DIR)

    for arch_name, cfg in ARCHS.items():
        out_path = CKPT_DIR / f"{arch_name}_fp32_extra_seed42.pt"
        if out_path.exists():
            print(f"\n=== {arch_name}: already done, skipping ===")
            continue

        print(f"\n=== {arch_name} ===")
        set_seed(FINE_TUNE_SEED)
        transform = get_transform(mean=cfg["mean"], std=cfg["std"])
        train_ds = GTSRBDataset(train_split_df, transform=transform)
        val_ds = GTSRBDataset(val_split_df, transform=transform)
        test_ds = GTSRBDataset(test_df, transform=transform)

        train_loader = DataLoader(train_ds, batch_size=FINE_TUNE_BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=FINE_TUNE_BATCH_SIZE, shuffle=False, num_workers=0)

        model = cfg["model_fn"]()
        model.load_state_dict(torch.load(CKPT_DIR / cfg["ckpt"], map_location="cpu"))
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
        best_val_f1, best_state = -1.0, None
        for epoch in range(FINE_TUNE_EPOCHS):
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
        torch.save(model.state_dict(), out_path)
        print(f"Saved FP32 extra-training state_dict to {out_path}")

        preds, labels = predict(model, test_ds, device, num_workers=0)
        results = summarize(preds, labels)
        print(f"{arch_name} FP32 extra-training test accuracy: {results['accuracy']:.4f}, macro-F1: {results['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
