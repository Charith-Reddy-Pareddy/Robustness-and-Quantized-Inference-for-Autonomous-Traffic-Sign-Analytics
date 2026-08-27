"""Latency and model-size benchmark for QAT-converted (real INT8) models, PyTorch/qnnpack
on CPU. Not directly comparable to benchmark_latency.py's ONNX Runtime numbers (different
runtime/backend) -- the meaningful comparison is FP32-vs-QAT-INT8 *within* this script,
each measured through the same PyTorch/CPU path.
"""

import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.baseline_cnn import BaselineCNN
from src.models.qat import load_converted
from src.models.transfer_model import build_mobilenet

CKPT_DIR = ROOT / "checkpoints"
REPORTS_DIR = ROOT / "reports"

N_WARMUP = 10
N_RUNS = 200

ARCHS = {
    "baseline_cnn": lambda: BaselineCNN(num_classes=43),
    "mobilenet_transfer": lambda: build_mobilenet(num_classes=43),
}


def model_size_kb(model: torch.nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(model.state_dict(), f.name)
        return Path(f.name).stat().st_size / 1024


def benchmark(model: torch.nn.Module) -> dict:
    model.eval()
    x = torch.rand(1, 3, 64, 64)

    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(x)

        latencies_ms = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            model(x)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": statistics.mean(latencies_ms),
        "std_ms": statistics.stdev(latencies_ms),
        "p50_ms": statistics.median(latencies_ms),
        "p95_ms": sorted(latencies_ms)[int(0.95 * N_RUNS)],
        "size_kb": model_size_kb(model),
    }


def main():
    results = {}
    for arch_name, fp32_fn in ARCHS.items():
        fp32_model = fp32_fn()
        fp32_model.load_state_dict(torch.load(CKPT_DIR / f"{arch_name}_seed42.pt", map_location="cpu"))
        fp32_stats = benchmark(fp32_model)

        qat_model = load_converted(arch_name, CKPT_DIR / f"{arch_name}_qat_seed42_converted.pt")
        qat_stats = benchmark(qat_model)

        results[arch_name] = {"fp32": fp32_stats, "qat_int8": qat_stats}
        speedup = fp32_stats["mean_ms"] / qat_stats["mean_ms"]
        shrink = fp32_stats["size_kb"] / qat_stats["size_kb"]
        print(f"{arch_name}:")
        print(f"  FP32:     {fp32_stats['mean_ms']:.3f}ms +/- {fp32_stats['std_ms']:.3f}ms, {fp32_stats['size_kb']:.1f} KB")
        print(f"  QAT INT8: {qat_stats['mean_ms']:.3f}ms +/- {qat_stats['std_ms']:.3f}ms, {qat_stats['size_kb']:.1f} KB")
        print(f"  -> {speedup:.2f}x latency, {shrink:.2f}x smaller")

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "latency_benchmark_qat.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
