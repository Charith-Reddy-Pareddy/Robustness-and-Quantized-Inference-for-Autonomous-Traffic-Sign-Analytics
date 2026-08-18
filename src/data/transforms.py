"""Shared preprocessing for GTSRB.

No train-time augmentation (rotation/brightness/noise/etc.) is applied on purpose:
those exact perturbations are used later as robustness *test* corruptions, and training
on them would confound "robust because of architecture/quantization" with "robust
because the model already saw this perturbation family during training."
"""

import torchvision.transforms as T

IMAGE_SIZE = 64
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

# MobileNetV2's pretrained ImageNet weights expect this specific normalization.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transform(image_size: int = IMAGE_SIZE, mean=NORM_MEAN, std=NORM_STD, corruption_fn=None) -> T.Compose:
    """corruption_fn, if given, is applied to the resized PIL image before ToTensor/Normalize
    so corruption severity is independent of any model's normalization scheme."""
    steps = [T.Resize((image_size, image_size))]
    if corruption_fn is not None:
        steps.append(T.Lambda(corruption_fn))
    steps += [T.ToTensor(), T.Normalize(mean=mean, std=std)]
    return T.Compose(steps)
