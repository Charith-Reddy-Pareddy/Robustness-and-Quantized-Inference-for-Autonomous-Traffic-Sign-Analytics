"""Quick EDA over GTSRB: class balance, image size distribution, split sanity check."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ingest import assert_no_track_leakage, load_train_dataframe, track_aware_split

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    df = load_train_dataframe(RAW_DIR)
    print(f"Total training images: {len(df)}")
    print(f"Classes: {df['ClassId'].nunique()}")
    print(f"Unique (class, track) groups: {df.groupby(['ClassId', 'TrackId']).ngroups}")

    train_df, val_df = track_aware_split(df, val_fraction=0.15, seed=42)
    assert_no_track_leakage(train_df, val_df)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  (no track leakage confirmed)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df["ClassId"].value_counts().sort_index().plot(kind="bar", ax=axes[0], width=0.9)
    axes[0].set_title("Class distribution (full train set)")
    axes[0].set_xlabel("ClassId")
    axes[0].set_ylabel("count")

    (df["Width"] * df["Height"]).apply(lambda a: a**0.5).hist(bins=40, ax=axes[1])
    axes[1].set_title("Image size distribution (sqrt(W*H))")
    axes[1].set_xlabel("pixels")

    fig.tight_layout()
    out_path = REPORTS_DIR / "eda_overview.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
