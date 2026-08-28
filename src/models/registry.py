"""Shared architecture registry.

Every evaluation/quantization script needs the same three things per architecture: how
to build a fresh model, and its normalization mean/std (baseline CNN and MobileNetV2
use different normalization since one trains from scratch and one is ImageNet-pretrained
-- see src/data/transforms.py). That triple was copy-pasted as its own `ARCHS = {...}`
dict literal in ~16 scripts; centralizing it here means a change to how a model is built
only needs to happen in one place.

scripts/run_experiments.py intentionally does NOT use this: its per-arch dict also
carries training hyperparameters (lr, epochs) that are a training-time concern, not an
architecture-identity one, and don't belong in a shared registry.
"""

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, NORM_MEAN, NORM_STD
from src.models.baseline_cnn import BaselineCNN
from src.models.transfer_model import build_mobilenet

NUM_CLASSES = 43

ARCH_SPECS = {
    "baseline_cnn": {
        "model_fn": lambda: BaselineCNN(num_classes=NUM_CLASSES),
        "mean": NORM_MEAN,
        "std": NORM_STD,
    },
    "mobilenet_transfer": {
        "model_fn": lambda: build_mobilenet(num_classes=NUM_CLASSES),
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
}

ARCH_NAMES = tuple(ARCH_SPECS.keys())


def ckpt_filename(arch: str, seed: int = 42) -> str:
    return f"{arch}_seed{seed}.pt"


def archs_with_ckpt(seed: int = 42) -> dict:
    """ARCH_SPECS with a "ckpt" filename added for the given seed -- the shape most
    single-seed eval scripts expect (they still do the actual torch.load themselves,
    since some load onto a specific device or wrap in NormalizedModel differently)."""
    return {name: {**spec, "ckpt": ckpt_filename(name, seed)} for name, spec in ARCH_SPECS.items()}
