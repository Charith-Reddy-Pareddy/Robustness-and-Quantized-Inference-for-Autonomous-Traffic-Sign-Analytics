"""GTSRB data ingestion: load metadata and build a track-aware train/val split.

GTSRB training images are sequential video frames of the same physical sign, encoded
in the filename as ``<classId>_<trackId>_<frameId>.png``. Splitting by image at random
leaks near-duplicate frames of the same sign across train/val, inflating validation
accuracy. We split by (ClassId, TrackId) group instead, so every frame of a given
physical sign stays on one side of the split.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

TRACK_ID_PATTERN = r"_(\d+)_\d+\.png$"


def load_train_dataframe(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "Train.csv")
    df["TrackId"] = df["Path"].str.extract(TRACK_ID_PATTERN).astype(int)
    df["Path"] = df["Path"].apply(lambda p: str(raw_dir / p))
    return df


def load_test_dataframe(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "Test.csv")
    df["Path"] = df["Path"].apply(lambda p: str(raw_dir / p))
    return df


def track_aware_split(df: pd.DataFrame, val_fraction: float = 0.15, seed: int = 42):
    """Split df into train/val by (ClassId, TrackId) group, stratified by class."""
    n_splits = max(round(1 / val_fraction), 2)
    groups = df["ClassId"].astype(str) + "_" + df["TrackId"].astype(str)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, df["ClassId"], groups))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def assert_no_track_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    train_groups = set(zip(train_df["ClassId"], train_df["TrackId"]))
    val_groups = set(zip(val_df["ClassId"], val_df["TrackId"]))
    overlap = train_groups & val_groups
    if overlap:
        raise ValueError(f"{len(overlap)} (class, track) groups leak across train/val")
