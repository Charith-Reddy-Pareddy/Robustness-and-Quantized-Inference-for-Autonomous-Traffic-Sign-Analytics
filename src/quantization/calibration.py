"""Calibration-set construction for static PTQ, shared by every quantize_* script."""

import numpy as np
import pandas as pd
import torch

from src.data.dataset import GTSRBDataset
from src.data.transforms import get_transform


def build_calibration_samples(train_df: pd.DataFrame, n: int, seed: int = 42) -> np.ndarray:
    """Sample `n` training images in raw pixel space (no normalization) for ONNX Runtime's
    static quantization calibrator, which expects the same preprocessing the exported graph
    applies internally."""
    sample_df = train_df.sample(n, random_state=seed)
    pixel_transform = get_transform(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ds = GTSRBDataset(sample_df, transform=pixel_transform)
    batch = torch.stack([ds[i][0] for i in range(len(ds))])
    return batch.numpy().astype(np.float32)
