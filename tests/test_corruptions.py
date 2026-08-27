import numpy as np
from PIL import Image

from src.robustness.corruptions import (
    CORRUPTION_CATEGORIES,
    CORRUPTIONS,
    SEVERITIES,
    brightness_contrast,
    fog,
    gaussian_blur,
    gaussian_noise,
    jpeg_compression,
    motion_blur,
    perspective,
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


def test_higher_severity_jpeg_compression_degrades_more():
    img = _sample_image()
    arr = np.asarray(img).astype(np.float32)
    low = np.asarray(jpeg_compression(1)(img)).astype(np.float32)
    high = np.asarray(jpeg_compression(4)(img)).astype(np.float32)
    assert np.abs(high - arr).mean() > np.abs(low - arr).mean()


def test_higher_severity_motion_blur_is_smoother_horizontally():
    img = _sample_image()
    low = np.asarray(motion_blur(1)(img)).astype(np.float32)
    high = np.asarray(motion_blur(4)(img)).astype(np.float32)
    # motion_blur is horizontal, so it smooths along the row (axis=1) specifically
    assert np.abs(np.diff(high, axis=1)).mean() < np.abs(np.diff(low, axis=1)).mean()


def test_motion_blur_is_directional_not_isotropic():
    """Distinguishes MotionBlur from GaussianBlur: it should smooth much more along
    rows than down columns, since the kernel is a horizontal line."""
    img = _sample_image()
    out = np.asarray(motion_blur(4)(img)).astype(np.float32)
    row_smoothness = np.abs(np.diff(out, axis=1)).mean()
    col_smoothness = np.abs(np.diff(out, axis=0)).mean()
    assert row_smoothness < col_smoothness


def test_higher_severity_fog_shifts_toward_fog_color():
    img = _sample_image()
    fog_color = np.array(fog(1).FOG_COLOR, dtype=np.float32)
    low = np.asarray(fog(1)(img)).astype(np.float32)
    high = np.asarray(fog(4)(img)).astype(np.float32)
    assert np.abs(high - fog_color).mean() < np.abs(low - fog_color).mean()


def test_higher_severity_perspective_exposes_more_fill_border():
    """On a random-noise image, mean pixel diff after warping is dominated by content
    misalignment rather than shift magnitude, so this checks something more direct:
    a bigger corner shift crops in more of the (128,128,128) fill border."""
    img = _sample_image()
    fill = np.array([128, 128, 128])

    def fill_fraction(out_arr):
        return np.all(out_arr == fill, axis=-1).mean()

    np.random.seed(0)
    small = np.asarray(perspective(1)(img))
    np.random.seed(0)
    large = np.asarray(perspective(4)(img))
    assert fill_fraction(large) > fill_fraction(small)


def test_corruption_categories_cover_every_registered_corruption():
    categorized = {name for names in CORRUPTION_CATEGORIES.values() for name in names}
    assert categorized == set(CORRUPTIONS.keys())
