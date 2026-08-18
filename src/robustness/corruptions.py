"""Controlled image corruptions for robustness testing, at 4 increasing severities.

Each factory takes a severity in {1,2,3,4} and returns a callable PIL.Image -> PIL.Image,
applied after resize and before ToTensor/Normalize (see src/data/transforms.get_transform).
Implemented as picklable classes (not closures) so they survive multiprocess DataLoader workers.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

BLUR_RADII = [0.5, 1.0, 1.75, 2.75]
NOISE_STDS = [0.05, 0.10, 0.16, 0.25]
ROTATION_DEGREES = [5, 10, 17, 25]
BRIGHTNESS_FACTORS = [0.8, 0.6, 0.4, 0.25]
CONTRAST_FACTORS = [0.9, 0.75, 0.6, 0.45]


class GaussianBlur:
    def __init__(self, severity: int):
        self.radius = BLUR_RADII[severity - 1]

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))


class GaussianNoise:
    def __init__(self, severity: int):
        self.std = NOISE_STDS[severity - 1]

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img).astype(np.float32) / 255.0
        noisy = arr + np.random.normal(0.0, self.std, arr.shape)
        noisy = np.clip(noisy, 0.0, 1.0) * 255.0
        return Image.fromarray(noisy.astype(np.uint8))


class Rotation:
    def __init__(self, severity: int):
        self.degrees = ROTATION_DEGREES[severity - 1]

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.rotate(self.degrees, resample=Image.BILINEAR, fillcolor=(128, 128, 128))


class BrightnessContrast:
    """Simulates dim/hazy lighting (dusk, overcast, glare) — darker and lower-contrast,
    a more realistic autonomous-deployment failure mode than pure rotation."""

    def __init__(self, severity: int):
        self.b_factor = BRIGHTNESS_FACTORS[severity - 1]
        self.c_factor = CONTRAST_FACTORS[severity - 1]

    def __call__(self, img: Image.Image) -> Image.Image:
        img = ImageEnhance.Brightness(img).enhance(self.b_factor)
        img = ImageEnhance.Contrast(img).enhance(self.c_factor)
        return img


# Kept as lowercase names for call-site readability: CORRUPTIONS["blur"](severity) -> callable
gaussian_blur = GaussianBlur
gaussian_noise = GaussianNoise
rotation = Rotation
brightness_contrast = BrightnessContrast

CORRUPTIONS = {
    "blur": GaussianBlur,
    "noise": GaussianNoise,
    "rotation": Rotation,
    "brightness_contrast": BrightnessContrast,
}
SEVERITIES = [1, 2, 3, 4]
