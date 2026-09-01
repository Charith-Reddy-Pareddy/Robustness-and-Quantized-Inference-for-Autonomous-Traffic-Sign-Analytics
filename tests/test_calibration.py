import pandas as pd
from PIL import Image

from src.quantization.calibration import build_calibration_samples


def _make_train_df(tmp_path, n=10):
    rows = []
    for i in range(n):
        path = tmp_path / f"{i:05d}.ppm"
        # Distinct fill color per image so sampling a different subset actually changes the
        # returned array -- identical blank images would pass a shuffled-order test by accident.
        color = ((i * 17) % 256, (i * 41) % 256, (i * 89) % 256)
        Image.new("RGB", (40, 40), color=color).save(path)
        rows.append({"Path": str(path), "ClassId": i % 3, "Roi.X1": 0, "Roi.Y1": 0, "Roi.X2": 40, "Roi.Y2": 40})
    return pd.DataFrame(rows)


def test_build_calibration_samples_returns_n_float32_images(tmp_path):
    train_df = _make_train_df(tmp_path)
    samples = build_calibration_samples(train_df, n=4)

    assert samples.shape == (4, 3, 64, 64)
    assert samples.dtype.name == "float32"


def test_build_calibration_samples_is_deterministic_per_seed(tmp_path):
    train_df = _make_train_df(tmp_path)
    a = build_calibration_samples(train_df, n=5, seed=1)
    b = build_calibration_samples(train_df, n=5, seed=1)
    c = build_calibration_samples(train_df, n=5, seed=2)

    assert (a == b).all()
    assert not (a == c).all()
