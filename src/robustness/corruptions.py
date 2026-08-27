"""Controlled image corruptions for robustness testing, at 4 increasing severities.

Each factory takes a severity in {1,2,3,4} and returns a callable PIL.Image -> PIL.Image,
applied after resize and before ToTensor/Normalize (see src/data/transforms.get_transform).
Implemented as picklable classes (not closures) so they survive multiprocess DataLoader workers.

Categorized (see CORRUPTION_CATEGORIES below), loosely following the noise/blur/weather/
digital split common in corruption-robustness benchmarks, adapted for a traffic-sign
deployment setting:
  - noise: sensor noise
  - blur: optical defocus, camera/subject motion
  - geometric: viewpoint change (rotation, perspective)
  - photometric: exposure/compression, not weather-caused
  - environmental: weather/scene effects a camera can't correct for
"""

import io

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

BLUR_RADII = [0.5, 1.0, 1.75, 2.75]
NOISE_STDS = [0.05, 0.10, 0.16, 0.25]
ROTATION_DEGREES = [5, 10, 17, 25]
BRIGHTNESS_FACTORS = [0.8, 0.6, 0.4, 0.25]
CONTRAST_FACTORS = [0.9, 0.75, 0.6, 0.45]
JPEG_QUALITIES = [70, 40, 20, 8]
MOTION_KERNEL_SIZES = [5, 9, 13, 19]
FOG_ALPHAS = [0.15, 0.30, 0.45, 0.60]
RAIN_STREAK_COUNTS = [15, 35, 60, 90]
SHADOW_FACTORS = [0.75, 0.6, 0.45, 0.3]
PERSPECTIVE_SHIFT_FRACTIONS = [0.06, 0.10, 0.15, 0.22]


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


class JpegCompression:
    """Low-quality JPEG re-encoding -- a realistic artifact of compressed camera feeds
    or transmitted imagery, distinct from anything caused by weather or optics."""

    def __init__(self, severity: int):
        self.quality = JPEG_QUALITIES[severity - 1]

    def __call__(self, img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class MotionBlur:
    """Directional (horizontal) blur, simulating vehicle motion during exposure --
    distinct from the isotropic GaussianBlur, which models optical defocus instead."""

    def __init__(self, severity: int):
        self.kernel_size = MOTION_KERNEL_SIZES[severity - 1]

    def __call__(self, img: Image.Image) -> Image.Image:
        kernel = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float32)
        kernel[self.kernel_size // 2, :] = 1.0 / self.kernel_size
        arr = np.asarray(img)
        blurred = cv2.filter2D(arr, -1, kernel)
        return Image.fromarray(blurred)


class Fog:
    """Alpha-blends a uniform haze layer over the image, reducing contrast and
    saturating toward a flat gray -- distinct from BrightnessContrast, which dims and
    flattens the *existing* pixels rather than blending in a new light source."""

    FOG_COLOR = (200, 200, 205)

    def __init__(self, severity: int):
        self.alpha = FOG_ALPHAS[severity - 1]

    def __call__(self, img: Image.Image) -> Image.Image:
        fog_layer = Image.new("RGB", img.size, self.FOG_COLOR)
        return Image.blend(img.convert("RGB"), fog_layer, self.alpha)


class Rain:
    """Overlays semi-transparent diagonal streaks. Density and opacity grow with
    severity; streak geometry (angle, length) is fixed so severity controls only how
    much rain is present, not its character."""

    STREAK_LENGTH = 14
    ANGLE_DX_DY = (3, 10)  # slightly off-vertical, a common rain-streak angle

    def __init__(self, severity: int):
        self.count = RAIN_STREAK_COUNTS[severity - 1]

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = img.size
        dx, dy = self.ANGLE_DX_DY
        norm = (dx**2 + dy**2) ** 0.5
        dx, dy = dx / norm * self.STREAK_LENGTH, dy / norm * self.STREAK_LENGTH
        xs = np.random.uniform(0, w, size=self.count)
        ys = np.random.uniform(0, h, size=self.count)
        for x, y in zip(xs, ys):
            draw.line([(x, y), (x + dx, y + dy)], fill=(210, 220, 235, 140), width=1)
        return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


class Shadow:
    """Darkens a random half-plane of the image, simulating a cast shadow (overpass,
    tree, another vehicle) falling across part of the sign."""

    def __init__(self, severity: int):
        self.factor = SHADOW_FACTORS[severity - 1]

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size
        mask = Image.new("L", img.size, 255)
        draw = ImageDraw.Draw(mask)
        # A shadow edge as a random line from one border to another, darkening
        # whichever side of it contains a randomly chosen anchor corner.
        edge_x, edge_y = np.random.uniform(0, w), np.random.uniform(0, h)
        angle = np.random.uniform(0, np.pi)
        length = max(w, h) * 2
        x0, y0 = edge_x - length * np.cos(angle), edge_y - length * np.sin(angle)
        x1, y1 = edge_x + length * np.cos(angle), edge_y + length * np.sin(angle)
        side_point = (0, 0) if np.random.rand() < 0.5 else (w, h)
        cross = (x1 - x0) * (side_point[1] - y0) - (y1 - y0) * (side_point[0] - x0)
        polygon = [(x0, y0), (x1, y1)]
        # Extend the cut line into a filled polygon covering one side of the image.
        corners = [(0, 0), (w, 0), (w, h), (0, h)]
        on_dark_side = [
            c for c in corners
            if ((x1 - x0) * (c[1] - y0) - (y1 - y0) * (c[0] - x0)) * cross >= 0
        ]
        draw.polygon(polygon + on_dark_side, fill=int(255 * self.factor))
        arr = np.asarray(img).astype(np.float32)
        mask_arr = np.asarray(mask).astype(np.float32) / 255.0
        darkened = arr * mask_arr[..., None]
        return Image.fromarray(darkened.astype(np.uint8))


class Perspective:
    """Warps the image as if the sign were viewed off-axis -- a genuinely different
    geometric distortion from Rotation (in-plane) since it changes apparent shape."""

    def __init__(self, severity: int):
        self.shift_fraction = PERSPECTIVE_SHIFT_FRACTIONS[severity - 1]

    @staticmethod
    def _find_coeffs(source_coords, target_coords):
        matrix = []
        for (x, y), (X, Y) in zip(source_coords, target_coords):
            matrix.append([X, Y, 1, 0, 0, 0, -x * X, -x * Y])
            matrix.append([0, 0, 0, X, Y, 1, -y * X, -y * Y])
        a = np.array(matrix, dtype=np.float64)
        b = np.array(source_coords, dtype=np.float64).reshape(8)
        return np.linalg.solve(a, b)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        shift = self.shift_fraction * min(w, h)
        original = [(0, 0), (w, 0), (w, h), (0, h)]
        distorted = [
            (np.random.uniform(0, shift), np.random.uniform(0, shift)),
            (w - np.random.uniform(0, shift), np.random.uniform(0, shift)),
            (w - np.random.uniform(0, shift), h - np.random.uniform(0, shift)),
            (np.random.uniform(0, shift), h - np.random.uniform(0, shift)),
        ]
        coeffs = self._find_coeffs(original, distorted)
        return img.convert("RGB").transform(
            (w, h), Image.PERSPECTIVE, coeffs, resample=Image.BILINEAR, fillcolor=(128, 128, 128)
        )


# Kept as lowercase names for call-site readability: CORRUPTIONS["blur"](severity) -> callable
gaussian_blur = GaussianBlur
gaussian_noise = GaussianNoise
rotation = Rotation
brightness_contrast = BrightnessContrast
jpeg_compression = JpegCompression
motion_blur = MotionBlur
fog = Fog
rain = Rain
shadow = Shadow
perspective = Perspective

CORRUPTIONS = {
    "blur": GaussianBlur,
    "noise": GaussianNoise,
    "rotation": Rotation,
    "brightness_contrast": BrightnessContrast,
    "jpeg_compression": JpegCompression,
    "motion_blur": MotionBlur,
    "fog": Fog,
    "rain": Rain,
    "shadow": Shadow,
    "perspective": Perspective,
}

CORRUPTION_CATEGORIES = {
    "noise": ["noise"],
    "blur": ["blur", "motion_blur"],
    "geometric": ["rotation", "perspective"],
    "photometric": ["brightness_contrast", "jpeg_compression"],
    "environmental": ["fog", "rain", "shadow"],
}

SEVERITIES = [1, 2, 3, 4]
