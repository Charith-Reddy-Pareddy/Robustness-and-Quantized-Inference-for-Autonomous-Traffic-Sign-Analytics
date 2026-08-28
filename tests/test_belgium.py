import pandas as pd
from PIL import Image

from src.data.belgium_ingest import load_belgium_dataframe
from src.data.belgium_mapping import GTSRB_TO_BELGIUM


def test_all_mapped_classes_are_valid_gtsrb_ids():
    for gtsrb_id in GTSRB_TO_BELGIUM:
        assert 0 <= gtsrb_id <= 42


def test_mapping_has_no_duplicate_belgium_classes():
    belgium_classes = list(GTSRB_TO_BELGIUM.values())
    assert len(belgium_classes) == len(set(belgium_classes))


def test_no_entry_is_deliberately_excluded():
    """GTSRB 17 ("no entry") has no confident BelgiumTSC match -- see belgium_mapping.py
    for why a plausible-looking candidate turned out to be a different sign."""
    assert 17 not in GTSRB_TO_BELGIUM


def _write_belgium_class(root, split_folder, belgium_class, rows):
    class_dir = root / split_folder / belgium_class
    class_dir.mkdir(parents=True)
    for i, (x1, y1, x2, y2) in enumerate(rows):
        fname = f"{i:05d}_00000.ppm"
        Image.new("RGB", (40, 40)).save(class_dir / fname)
    gt_rows = [
        {"Filename": f"{i:05d}_00000.ppm", "Width": 40, "Height": 40, "Roi.X1": x1, "Roi.Y1": y1, "Roi.X2": x2, "Roi.Y2": y2}
        for i, (x1, y1, x2, y2) in enumerate(rows)
    ]
    pd.DataFrame(gt_rows).to_csv(class_dir / f"GT-{belgium_class}.csv", sep=";", index=False)


def test_load_belgium_dataframe_assigns_correct_labels_and_roi(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.belgium_ingest.GTSRB_TO_BELGIUM", {14: "00022", 13: "00019"})
    _write_belgium_class(tmp_path, "BelgiumTSC_Training/Training", "00022", [(1, 2, 30, 31)])
    _write_belgium_class(tmp_path, "BelgiumTSC_Testing/Testing", "00022", [(3, 4, 32, 33)])
    _write_belgium_class(tmp_path, "BelgiumTSC_Training/Training", "00019", [(5, 6, 34, 35), (7, 8, 36, 37)])

    df = load_belgium_dataframe(tmp_path)

    assert len(df) == 4
    assert set(df["ClassId"]) == {14, 13}
    assert (df[df["ClassId"] == 14]["Path"].apply(lambda p: "00022" in p)).all()
    stop_row = df[(df["ClassId"] == 14) & (df["Path"].str.contains("Training"))].iloc[0]
    assert (stop_row["Roi.X1"], stop_row["Roi.Y1"], stop_row["Roi.X2"], stop_row["Roi.Y2"]) == (1, 2, 30, 31)


def test_load_belgium_dataframe_combines_training_and_testing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.belgium_ingest.GTSRB_TO_BELGIUM", {14: "00022"})
    _write_belgium_class(tmp_path, "BelgiumTSC_Training/Training", "00022", [(0, 0, 10, 10)])
    _write_belgium_class(tmp_path, "BelgiumTSC_Testing/Testing", "00022", [(0, 0, 10, 10), (0, 0, 10, 10)])

    df = load_belgium_dataframe(tmp_path)

    assert len(df) == 3


def test_load_belgium_dataframe_skips_missing_class_folders(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.belgium_ingest.GTSRB_TO_BELGIUM", {14: "00022"})
    (tmp_path / "BelgiumTSC_Training" / "Training").mkdir(parents=True)
    (tmp_path / "BelgiumTSC_Testing" / "Testing").mkdir(parents=True)

    df = load_belgium_dataframe(tmp_path)

    assert len(df) == 0
