"""Export a trained model (wrapped with its normalization) to a FP32 ONNX graph.

The graph's input is a raw [0,1] pixel tensor, not the model's normalized input — the
NormalizedModel wrapper bakes normalization into the exported graph. This lets every
downstream consumer (corruption tests, adversarial transfer attacks) feed the same
pixel-space tensors regardless of which architecture's ONNX graph they're calling,
exactly mirroring how the PyTorch-side evaluation scripts already work.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


def export_to_onnx(wrapped_model: torch.nn.Module, onnx_path: Path, image_size: int = 64) -> None:
    wrapped_model.eval()
    dummy = torch.rand(1, 3, image_size, image_size)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped_model,
        dummy,
        str(onnx_path),
        input_names=["pixel_image"],
        output_names=["logits"],
        dynamic_axes={"pixel_image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
        dynamo=False,
    )


def assert_onnx_matches_torch(wrapped_model: torch.nn.Module, onnx_path: Path, atol: float = 1e-4) -> float:
    wrapped_model.eval()
    dummy = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        torch_out = wrapped_model(dummy).numpy()
    sess = ort.InferenceSession(str(onnx_path))
    onnx_out = sess.run(None, {"pixel_image": dummy.numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    assert max_diff < atol, f"ONNX export diverges from PyTorch by {max_diff} (atol={atol})"
    return max_diff
