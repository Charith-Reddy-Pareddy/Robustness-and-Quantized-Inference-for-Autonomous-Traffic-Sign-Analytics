"""MobileNetV2 transfer-learning model: ImageNet backbone, new 43-class head."""

import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_mobilenet(num_classes: int = 43) -> nn.Module:
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
