"""Grad-CAM comparison grid: original image vs baseline CNN CAM vs MobileNetV2 CAM.

Required deliverable per spec — confirms both models attend to the sign itself rather
than background artifacts. Runs on CPU: Grad-CAM's backward pass over a handful of
images doesn't need MPS, and CPU avoids any MPS-autograd compatibility edge cases.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.ingest import load_test_dataframe
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, NORM_MEAN, NORM_STD, get_transform
from src.models.baseline_cnn import BaselineCNN
from src.models.transfer_model import build_mobilenet
from src.viz.gradcam import compute_cam_overlay, target_layer_for

RAW_DIR = ROOT / "data" / "raw"
CKPT_DIR = ROOT / "checkpoints"
REPORTS_DIR = ROOT / "reports"

DEVICE = torch.device("cpu")
N_SAMPLES = 6

ARCHS = {
    "baseline_cnn": {
        "model_fn": lambda: BaselineCNN(num_classes=43),
        "mean": NORM_MEAN,
        "std": NORM_STD,
        "ckpt": "baseline_cnn_seed42.pt",
    },
    "mobilenet_transfer": {
        "model_fn": lambda: build_mobilenet(num_classes=43),
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "ckpt": "mobilenet_transfer_seed42.pt",
    },
}


def load_model(arch_name: str):
    cfg = ARCHS[arch_name]
    model = cfg["model_fn"]()
    model.load_state_dict(torch.load(CKPT_DIR / cfg["ckpt"], map_location=DEVICE))
    return model.to(DEVICE).eval(), cfg


def main():
    test_df = load_test_dataframe(RAW_DIR)
    sample_df = test_df.groupby("ClassId").first().reset_index().sample(
        N_SAMPLES, random_state=7
    )

    unnorm_transform = get_transform(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    baseline_model, baseline_cfg = load_model("baseline_cnn")
    mobilenet_model, mobilenet_cfg = load_model("mobilenet_transfer")

    fig, axes = plt.subplots(N_SAMPLES, 3, figsize=(7, 2.3 * N_SAMPLES))
    for row, (_, sample) in enumerate(sample_df.iterrows()):
        from PIL import Image

        img = Image.open(sample["Path"]).convert("RGB")
        img = img.crop((sample["Roi.X1"], sample["Roi.Y1"], sample["Roi.X2"], sample["Roi.Y2"]))

        rgb_img = unnorm_transform(img).permute(1, 2, 0).numpy().astype(np.float32)

        overlays = {}
        for arch_name, (model, cfg) in [
            ("baseline_cnn", (baseline_model, baseline_cfg)),
            ("mobilenet_transfer", (mobilenet_model, mobilenet_cfg)),
        ]:
            input_tensor = get_transform(mean=cfg["mean"], std=cfg["std"])(img).unsqueeze(0)
            target_layers = target_layer_for(arch_name, model)
            overlays[arch_name] = compute_cam_overlay(model, target_layers, input_tensor, rgb_img)

        axes[row, 0].imshow(rgb_img)
        axes[row, 0].set_ylabel(f"class {sample['ClassId']}", fontsize=9)
        axes[row, 1].imshow(overlays["baseline_cnn"])
        axes[row, 2].imshow(overlays["mobilenet_transfer"])
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0, 0].set_title("Original")
    axes[0, 1].set_title("Baseline CNN\nGrad-CAM")
    axes[0, 2].set_title("MobileNetV2\nGrad-CAM")
    fig.tight_layout()
    out_path = REPORTS_DIR / "gradcam_grid.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
