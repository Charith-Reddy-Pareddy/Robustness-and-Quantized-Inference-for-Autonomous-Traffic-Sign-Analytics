"""Wraps a model with its normalization so adversarial attacks can operate directly in raw
[0,1] pixel space. Without this, the same epsilon would mean a different amount of visible
perturbation for the baseline CNN (std=0.5) vs MobileNetV2 (ImageNet std) — this wrapper
makes epsilon comparable across architectures, and this pixel-space convention is what lets
Phase 2 (INT8 transfer attacks) reuse the same perturbations unchanged.
"""

import torch
import torch.nn as nn


class NormalizedModel(nn.Module):
    def __init__(self, model: nn.Module, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x_pixel: torch.Tensor) -> torch.Tensor:
        return self.model((x_pixel - self.mean) / self.std)
