"""Quantize seed-42 models at multiple calibration-set sizes, for the calibration-size
ablation (how much calibration data does static PTQ actually need?).
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.ingest import load_train_dataframe
from src.models.registry import archs_with_ckpt
from src.models.wrapper import NormalizedModel
from src.quantization.calibration import build_calibration_samples
from src.quantization.export import export_to_onnx
from src.quantization.quantize import quantize_to_int8

RAW_DIR = ROOT / "data" / "raw"
CKPT_DIR = ROOT / "checkpoints"
ONNX_DIR = ROOT / "onnx" / "calibration_ablation"

CALIBRATION_SIZES = [50, 100, 200, 500, 1000, 2000]

ARCHS = archs_with_ckpt()


def main():
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    train_df = load_train_dataframe(RAW_DIR)

    for arch_name, cfg in ARCHS.items():
        base_model = cfg["model_fn"]()
        base_model.load_state_dict(torch.load(CKPT_DIR / cfg["ckpt"], map_location="cpu"))
        wrapped = NormalizedModel(base_model, cfg["mean"], cfg["std"]).eval()

        fp32_path = ONNX_DIR / f"{arch_name}_fp32.onnx"
        export_to_onnx(wrapped, fp32_path)

        for n in CALIBRATION_SIZES:
            calibration_samples = build_calibration_samples(train_df, n)
            int8_path = ONNX_DIR / f"{arch_name}_calib{n}_int8.onnx"
            quantize_to_int8(fp32_path, int8_path, calibration_samples)
            print(f"{arch_name}: quantized with {n}-image calibration set")


if __name__ == "__main__":
    main()
