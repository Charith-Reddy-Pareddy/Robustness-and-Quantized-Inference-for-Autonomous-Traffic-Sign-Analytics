"""Loads the Mapillary+DFG crops for the classes mapped to a GTSRB class ID (see
mapillary_mapping.py) into a dataframe compatible with GTSRBDataset (crop_to_roi=False,
since these images are already pre-cropped to the sign)."""

import random
from pathlib import Path

import pandas as pd

from src.data.mapillary_mapping import GTSRB_TO_MAPILLARY


def load_mapillary_dataframe(mapillary_dir: Path, samples_per_class: int | None = None, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for gtsrb_id, mapillary_class in GTSRB_TO_MAPILLARY.items():
        class_dir = mapillary_dir / "crops" / mapillary_class
        paths = sorted(class_dir.glob("*.jpg"))
        if samples_per_class is not None and len(paths) > samples_per_class:
            paths = rng.sample(paths, samples_per_class)
        rows.extend({"Path": str(p), "ClassId": gtsrb_id} for p in paths)
    return pd.DataFrame(rows)
