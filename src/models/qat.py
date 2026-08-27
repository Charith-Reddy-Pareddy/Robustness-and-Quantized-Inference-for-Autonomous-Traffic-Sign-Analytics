"""Quantization-aware training (QAT) model builders.

Uses PyTorch's eager-mode QAT API (fake-quantization inserted during training via
observers with a straight-through estimator, so gradients flow normally) rather than
ONNX Runtime's post-training static quantization (src/quantization/) used elsewhere in
this project. The two are genuinely different quantization pipelines being compared, not
just two code paths to the same result.

Backend is fixed to qnnpack (the only quantized engine available on Apple Silicon; fbgemm
is x86-only). The *prepared* (pre-convert) QAT model stays differentiable throughout
training and is exactly the "quantization-aware differentiable surrogate" that
reports/ROBUSTNESS_REPORT.md's Threat Models section identifies as missing for a true
INT8 white-box attack -- see scripts/eval_qat.py.
"""

import torch
import torch.nn as nn
from torch.ao.quantization import DeQuantStub, QuantStub, convert, get_default_qat_qconfig, prepare_qat
from torchvision.models.quantization import mobilenet_v2 as quantizable_mobilenet_v2

from src.models.baseline_cnn import BaselineCNN

QAT_BACKEND = "qnnpack"

BASELINE_CNN_FUSE_GROUPS = [
    ["features.0", "features.1", "features.2"],
    ["features.3", "features.4", "features.5"],
    ["features.7", "features.8", "features.9"],
    ["features.10", "features.11", "features.12"],
    ["features.14", "features.15", "features.16"],
]


class BaselineCNNQAT(BaselineCNN):
    """BaselineCNN with Quant/DeQuant stubs and layer fusion for QAT.

    Subclasses BaselineCNN (rather than wrapping it) so its state_dict keys match the
    FP32-trained checkpoint exactly -- QAT fine-tuning starts from those trained weights,
    not from scratch.
    """

    def __init__(self, num_classes: int = 43):
        super().__init__(num_classes)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        return self.dequant(x)

    def fuse_model(self, is_qat: bool = True) -> None:
        torch.ao.quantization.fuse_modules(self, BASELINE_CNN_FUSE_GROUPS, inplace=True)


def build_baseline_cnn_qat(num_classes: int = 43) -> BaselineCNNQAT:
    return BaselineCNNQAT(num_classes)


def build_mobilenet_qat(num_classes: int = 43) -> nn.Module:
    model = quantizable_mobilenet_v2(weights=None, quantize=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def prepare_for_qat(model: nn.Module) -> nn.Module:
    """Fuse conv-bn-relu triples, attach a QAT qconfig, and insert fake-quant observers.

    Returns the same model object (mutated in place) in train() mode, ready to fine-tune.
    """
    torch.backends.quantized.engine = QAT_BACKEND
    model.eval()
    model.fuse_model(is_qat=True)
    model.train()
    model.qconfig = get_default_qat_qconfig(QAT_BACKEND)
    prepare_qat(model, inplace=True)
    return model


def convert_to_quantized(model: nn.Module) -> nn.Module:
    """Convert a fine-tuned, fake-quantized (prepared) model to a real INT8 model.

    The input model must already be prepare_for_qat'd and fine-tuned; this returns a new
    module (torch.ao.quantization.convert does not mutate reliably in place for all
    module types) that runs true INT8 ops on CPU via the qnnpack backend.
    """
    torch.backends.quantized.engine = QAT_BACKEND
    model.eval()
    return convert(model, inplace=False)


BUILD_FNS = {"baseline_cnn": build_baseline_cnn_qat, "mobilenet_transfer": build_mobilenet_qat}


def load_prepared(arch_name: str, state_dict_path) -> nn.Module:
    """Rebuild a prepared (fake-quant, differentiable) QAT model and load its weights.

    Whole-module pickling doesn't work for prepared models: prepare_qat attaches a
    qconfig containing an unpicklable local closure. State_dicts only contain tensors, so
    saving/loading those instead sidesteps the problem -- but the receiving process must
    rebuild the exact same fuse+qconfig+prepare_qat structure first, since that's what
    determines the state_dict's key names and shapes.
    """
    model = BUILD_FNS[arch_name](num_classes=43)
    model = prepare_for_qat(model)
    model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    return model


def load_converted(arch_name: str, state_dict_path) -> nn.Module:
    """Rebuild a converted (real INT8) QAT model and load its weights. See load_prepared."""
    model = BUILD_FNS[arch_name](num_classes=43)
    model = prepare_for_qat(model)
    converted = convert_to_quantized(model)
    converted.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    return converted
