import numpy as np
from PIL import Image

from src.robustness.corruptions import (
    CORRUPTIONS,
    SEVERITIES,
    brightness_contrast,
    gaussian_blur,
    gaussian_noise,
    rotation,
)


def _sample_image():
    arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def test_all_corruptions_preserve_image_size_and_mode():
    img = _sample_image()
    for factory in CORRUPTIONS.values():
        for severity in SEVERITIES:
            out = factory(severity)(img)
            assert out.size == img.size
            assert out.mode == "RGB"


def test_higher_severity_noise_perturbs_more():
    img = _sample_image()
    arr = np.asarray(img).astype(np.float32)
    low = np.asarray(gaussian_noise(1)(img)).astype(np.float32)
    high = np.asarray(gaussian_noise(4)(img)).astype(np.float32)
    assert np.abs(high - arr).mean() > np.abs(low - arr).mean()


def test_higher_severity_blur_is_smoother():
    img = _sample_image()
    low = np.asarray(gaussian_blur(1)(img)).astype(np.float32)
    high = np.asarray(gaussian_blur(4)(img)).astype(np.float32)
    # more blur -> lower local variance (gradient magnitude)
    assert np.abs(np.diff(high, axis=0)).mean() < np.abs(np.diff(low, axis=0)).mean()


def test_higher_severity_brightness_contrast_darkens():
    img = _sample_image()
    low = np.asarray(brightness_contrast(1)(img)).astype(np.float32)
    high = np.asarray(brightness_contrast(4)(img)).astype(np.float32)
    assert high.mean() < low.mean()


def test_rotation_changes_pixels_by_increasing_amount():
    img = _sample_image()
    arr = np.asarray(img).astype(np.float32)
    small = np.asarray(rotation(1)(img)).astype(np.float32)
    large = np.asarray(rotation(4)(img)).astype(np.float32)
    assert np.abs(large - arr).mean() > np.abs(small - arr).mean()
