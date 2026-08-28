"""Export + statically quantize all 5 seeds (not just seed 42) for the multi-seed
robustness variance study. Same calibration set/settings as quantize_models.py.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import GTSRBDataset
from src.data.ingest import load_train_dataframe
from src.data.transforms import get_transform
from src.models.registry import ARCH_SPECS
from src.models.wrapper import NormalizedModel
from src.quantization.export import assert_onnx_matches_torch, export_to_onnx
from src.quantization.quantize import quantize_to_int8

RAW_DIR = ROOT / "data" / "raw"
CKPT_DIR = ROOT / "checkpoints"
ONNX_DIR = ROOT / "onnx" / "multiseed"

N_CALIBRATION = 1000
SEEDS = [42, 123, 2024, 7, 999]

ARCHS = ARCH_SPECS


def build_calibration_samples(train_df, n: int) -> np.ndarray:
    sample_df = train_df.sample(n, random_state=42)
    pixel_transform = get_transform(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ds = GTSRBDataset(sample_df, transform=pixel_transform)
    batch = torch.stack([ds[i][0] for i in range(len(ds))])
    return batch.numpy().astype(np.float32)


def main():
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    train_df = load_train_dataframe(RAW_DIR)
    calibration_samples = build_calibration_samples(train_df, N_CALIBRATION)
    print(f"Calibration set: {calibration_samples.shape}")

    for arch_name, cfg in ARCHS.items():
        for seed in SEEDS:
            ckpt = CKPT_DIR / f"{arch_name}_seed{seed}.pt"
            base_model = cfg["model_fn"]()
            base_model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            wrapped = NormalizedModel(base_model, cfg["mean"], cfg["std"]).eval()

            fp32_path = ONNX_DIR / f"{arch_name}_seed{seed}_fp32.onnx"
            int8_path = ONNX_DIR / f"{arch_name}_seed{seed}_int8.onnx"

            export_to_onnx(wrapped, fp32_path)
            max_diff = assert_onnx_matches_torch(wrapped, fp32_path)
            quantize_to_int8(fp32_path, int8_path, calibration_samples)
            print(f"{arch_name} seed{seed}: exported + quantized (max diff {max_diff:.2e})")


if __name__ == "__main__":
    main()
