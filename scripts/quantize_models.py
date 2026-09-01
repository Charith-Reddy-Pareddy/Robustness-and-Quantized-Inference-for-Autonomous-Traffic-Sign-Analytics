"""Export both FP32 models to ONNX and produce statically-quantized INT8 versions."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.ingest import load_train_dataframe
from src.models.registry import archs_with_ckpt
from src.models.wrapper import NormalizedModel
from src.quantization.calibration import build_calibration_samples
from src.quantization.export import assert_onnx_matches_torch, export_to_onnx
from src.quantization.quantize import quantize_to_int8

RAW_DIR = ROOT / "data" / "raw"
CKPT_DIR = ROOT / "checkpoints"
ONNX_DIR = ROOT / "onnx"

N_CALIBRATION = 1000

ARCHS = archs_with_ckpt()


def main():
    ONNX_DIR.mkdir(exist_ok=True)
    train_df = load_train_dataframe(RAW_DIR)
    calibration_samples = build_calibration_samples(train_df, N_CALIBRATION)
    print(f"Calibration set: {calibration_samples.shape}")

    for arch_name, cfg in ARCHS.items():
        base_model = cfg["model_fn"]()
        base_model.load_state_dict(torch.load(CKPT_DIR / cfg["ckpt"], map_location="cpu"))
        wrapped = NormalizedModel(base_model, cfg["mean"], cfg["std"]).eval()

        fp32_path = ONNX_DIR / f"{arch_name}_fp32.onnx"
        int8_path = ONNX_DIR / f"{arch_name}_int8.onnx"

        export_to_onnx(wrapped, fp32_path)
        max_diff = assert_onnx_matches_torch(wrapped, fp32_path)
        print(f"{arch_name}: FP32 ONNX export matches PyTorch (max diff {max_diff:.2e})")

        quantize_to_int8(fp32_path, int8_path, calibration_samples)

        fp32_size = fp32_path.stat().st_size / 1024
        int8_size = int8_path.stat().st_size / 1024
        print(
            f"{arch_name}: FP32 size {fp32_size:.1f} KB, INT8 size {int8_size:.1f} KB "
            f"({fp32_size / int8_size:.2f}x smaller)"
        )


if __name__ == "__main__":
    main()
