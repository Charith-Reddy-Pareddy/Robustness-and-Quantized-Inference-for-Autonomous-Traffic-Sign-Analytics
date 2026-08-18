"""PyTorch Dataset for GTSRB."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class GTSRBDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, crop_to_roi: bool = True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.crop_to_roi = crop_to_roi

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["Path"]).convert("RGB")
        if self.crop_to_roi:
            image = image.crop((row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"]))
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(int(row["ClassId"]), dtype=torch.long)
        return image, label


def class_names(meta_csv: Path) -> dict[int, str]:
    """GTSRB's Meta.csv only has ClassId/Shape/etc, not human names, so fall back to the id."""
    meta = pd.read_csv(meta_csv)
    return {int(row.ClassId): str(int(row.ClassId)) for row in meta.itertuples()}
