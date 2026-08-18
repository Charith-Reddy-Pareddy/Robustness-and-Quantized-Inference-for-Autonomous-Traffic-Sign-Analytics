from PIL import Image

from src.data.mapillary_ingest import load_mapillary_dataframe
from src.data.mapillary_mapping import GTSRB_TO_MAPILLARY


def test_all_mapped_classes_are_valid_gtsrb_ids():
    for gtsrb_id in GTSRB_TO_MAPILLARY:
        assert 0 <= gtsrb_id <= 42


def test_mapping_has_no_duplicate_mapillary_classes():
    mapillary_classes = list(GTSRB_TO_MAPILLARY.values())
    assert len(mapillary_classes) == len(set(mapillary_classes))


def test_load_mapillary_dataframe_assigns_correct_labels(tmp_path):
    (tmp_path / "crops" / "regulatory--stop").mkdir(parents=True)
    (tmp_path / "crops" / "regulatory--yield").mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (20, 20)).save(tmp_path / "crops" / "regulatory--stop" / f"{i}.jpg")
    for i in range(2):
        Image.new("RGB", (20, 20)).save(tmp_path / "crops" / "regulatory--yield" / f"{i}.jpg")

    df = load_mapillary_dataframe(tmp_path)
    assert len(df) == 5
    assert (df[df["ClassId"] == 14]["Path"].apply(lambda p: "regulatory--stop" in p)).all()
    assert (df[df["ClassId"] == 13]["Path"].apply(lambda p: "regulatory--yield" in p)).all()


def test_load_mapillary_dataframe_respects_samples_per_class_cap(tmp_path):
    class_dir = tmp_path / "crops" / "regulatory--stop"
    class_dir.mkdir(parents=True)
    for i in range(10):
        Image.new("RGB", (20, 20)).save(class_dir / f"{i}.jpg")

    df = load_mapillary_dataframe(tmp_path, samples_per_class=4, seed=0)
    assert (df["ClassId"] == 14).sum() == 4


def test_load_mapillary_dataframe_skips_missing_class_dirs(tmp_path):
    (tmp_path / "crops").mkdir()
    df = load_mapillary_dataframe(tmp_path)
    assert len(df) == 0
