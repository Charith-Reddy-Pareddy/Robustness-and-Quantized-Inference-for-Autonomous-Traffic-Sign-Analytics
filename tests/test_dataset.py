import numpy as np
import pandas as pd
from PIL import Image

from src.data.dataset import GTSRBDataset


def _synthetic_image_with_marker(path, size=(40, 20), marker_box=(25, 2, 35, 8)):
    """A black image with a white rectangle at marker_box = (x1, y1, x2, y2)."""
    img = Image.new("RGB", size, (0, 0, 0))
    x1, y1, x2, y2 = marker_box
    for x in range(x1, x2):
        for y in range(y1, y2):
            img.putpixel((x, y), (255, 255, 255))
    img.save(path)
    return marker_box


def test_crop_uses_x_y_axes_in_correct_order(tmp_path):
    """Regression test for a bug where Roi.X1/Roi.Y1 were swapped in Image.crop(),
    which silently cropped the wrong region for any non-square ROI."""
    img_path = tmp_path / "marker.png"
    x1, y1, x2, y2 = _synthetic_image_with_marker(img_path)
    expected_width, expected_height = x2 - x1, y2 - y1

    df = pd.DataFrame(
        [{"Path": str(img_path), "ClassId": 0, "Roi.X1": x1, "Roi.Y1": y1, "Roi.X2": x2, "Roi.Y2": y2}]
    )
    ds = GTSRBDataset(df, transform=None)
    cropped, _ = ds[0]

    assert cropped.size == (expected_width, expected_height)
    arr = np.asarray(cropped)
    assert arr.mean() > 200  # the marker region is all-white; a wrong crop lands on black


def test_dataset_returns_correct_label(tmp_path):
    img_path = tmp_path / "img.png"
    Image.new("RGB", (20, 20), (0, 0, 0)).save(img_path)
    df = pd.DataFrame(
        [{"Path": str(img_path), "ClassId": 7, "Roi.X1": 0, "Roi.Y1": 0, "Roi.X2": 20, "Roi.Y2": 20}]
    )
    ds = GTSRBDataset(df, transform=None)
    _, label = ds[0]
    assert label.item() == 7


def test_crop_to_roi_false_returns_full_image(tmp_path):
    img_path = tmp_path / "img.png"
    Image.new("RGB", (30, 10), (0, 0, 0)).save(img_path)
    df = pd.DataFrame(
        [{"Path": str(img_path), "ClassId": 0, "Roi.X1": 5, "Roi.Y1": 1, "Roi.X2": 10, "Roi.Y2": 3}]
    )
    ds = GTSRBDataset(df, transform=None, crop_to_roi=False)
    image, _ = ds[0]
    assert image.size == (30, 10)
