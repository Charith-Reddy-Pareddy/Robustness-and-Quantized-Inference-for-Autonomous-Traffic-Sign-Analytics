"""Loads BelgiumTSC images for the classes mapped to a GTSRB class ID (see
belgium_mapping.py) into a dataframe compatible with GTSRBDataset. Unlike the
Mapillary+DFG loader, BelgiumTSC ships the same per-image Roi.X1/Y1/X2/Y2 + ClassId
annotation format as GTSRB itself (one GT-<class>.csv per class folder), so crop_to_roi
can stay True here, same as native GTSRB.

Combines the Training and Testing splits into a single pool: this is an evaluation-only
generalization check (no BelgiumTSC-side training happens anywhere in this project), and
several mapped classes have too few images in either split alone for a stable estimate.
"""

from pathlib import Path

import pandas as pd

from src.data.belgium_mapping import GTSRB_TO_BELGIUM


def load_belgium_dataframe(belgium_dir: Path) -> pd.DataFrame:
    frames = []
    for split, folder in [("Training", "BelgiumTSC_Training"), ("Testing", "BelgiumTSC_Testing")]:
        split_dir = belgium_dir / folder / split
        for gtsrb_id, belgium_class in GTSRB_TO_BELGIUM.items():
            class_dir = split_dir / belgium_class
            gt_csv = class_dir / f"GT-{belgium_class}.csv"
            if not gt_csv.exists():
                continue
            df = pd.read_csv(gt_csv, sep=";")
            df["Path"] = df["Filename"].apply(lambda f, d=class_dir: str(d / f))
            df["ClassId"] = gtsrb_id
            frames.append(df[["Path", "ClassId", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"]])
    if not frames:
        return pd.DataFrame(columns=["Path", "ClassId", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"])
    return pd.concat(frames, ignore_index=True)
