import pandas as pd

from src.data.ingest import assert_no_track_leakage, track_aware_split


def _synthetic_df(n_classes=5, tracks_per_class=25, frames_per_track=6):
    rows = []
    for class_id in range(n_classes):
        for track_id in range(tracks_per_class):
            for frame_id in range(frames_per_track):
                rows.append(
                    {
                        "ClassId": class_id,
                        "TrackId": track_id,
                        "Path": f"Train/{class_id}/{class_id:05d}_{track_id:05d}_{frame_id:05d}.png",
                        "Width": 32,
                        "Height": 32,
                        "Roi.X1": 0,
                        "Roi.Y1": 0,
                        "Roi.X2": 32,
                        "Roi.Y2": 32,
                    }
                )
    return pd.DataFrame(rows)


def test_split_has_no_track_leakage():
    df = _synthetic_df()
    train_df, val_df = track_aware_split(df, val_fraction=0.2, seed=0)
    assert_no_track_leakage(train_df, val_df)


def test_split_covers_all_rows_exactly_once():
    df = _synthetic_df()
    train_df, val_df = track_aware_split(df, val_fraction=0.2, seed=0)
    assert len(train_df) + len(val_df) == len(df)
    assert set(train_df["Path"]).isdisjoint(set(val_df["Path"]))


def test_split_is_roughly_stratified_by_class():
    df = _synthetic_df()
    train_df, val_df = track_aware_split(df, val_fraction=0.2, seed=0)
    train_counts = train_df["ClassId"].value_counts(normalize=True)
    val_counts = val_df["ClassId"].value_counts(normalize=True)
    for class_id in df["ClassId"].unique():
        assert abs(train_counts.get(class_id, 0) - val_counts.get(class_id, 0)) < 0.15


def test_all_val_classes_present_in_train():
    df = _synthetic_df()
    train_df, val_df = track_aware_split(df, val_fraction=0.2, seed=0)
    assert set(val_df["ClassId"]).issubset(set(train_df["ClassId"]))
