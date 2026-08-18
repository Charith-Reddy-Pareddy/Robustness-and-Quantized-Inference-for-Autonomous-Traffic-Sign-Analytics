"""Grad-CAM helpers: confirm each model attends to the sign, not background artifacts."""

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def target_layer_for(arch_name: str, model: torch.nn.Module):
    if arch_name == "baseline_cnn":
        return [model.features[16]]  # last conv block's ReLU, pre-final-maxpool (8x8 map)
    if arch_name == "mobilenet_transfer":
        # Not the final block: at 64x64 input MobileNetV2 downsamples to a 2x2 feature map
        # by the last layer, too coarse to localize anything. features[6] is the last block
        # at 8x8 resolution (matches the baseline CNN's CAM resolution) while still being
        # deep enough to carry semantic features.
        return [model.features[6]]
    raise ValueError(f"no known Grad-CAM target layer for arch '{arch_name}'")


def compute_cam_overlay(model, target_layers, input_tensor: torch.Tensor, rgb_img: np.ndarray) -> np.ndarray:
    """rgb_img must be float32 in [0,1], HxWx3 — unnormalized, for visualization only."""
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    return show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
