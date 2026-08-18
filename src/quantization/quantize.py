"""Static INT8 quantization of an exported ONNX graph, calibrated on real training images."""

from pathlib import Path

import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static


class ArrayCalibrationReader(CalibrationDataReader):
    """Feeds pre-loaded pixel-space numpy batches to the ONNX Runtime calibrator."""

    def __init__(self, samples: np.ndarray, input_name: str, batch_size: int = 8):
        self.input_name = input_name
        self.samples = samples
        self.batch_size = batch_size
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.samples):
            return None
        batch = self.samples[self.idx : self.idx + self.batch_size]
        self.idx += self.batch_size
        return {self.input_name: batch}


def quantize_to_int8(fp32_path: Path, int8_path: Path, calibration_samples: np.ndarray, input_name: str = "pixel_image") -> None:
    reader = ArrayCalibrationReader(calibration_samples, input_name)
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
