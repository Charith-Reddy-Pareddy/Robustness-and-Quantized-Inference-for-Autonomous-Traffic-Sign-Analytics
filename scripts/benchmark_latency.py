"""Latency and model-size benchmark: FP32 vs INT8, both via ONNX Runtime CPU execution.

Comparing both formats through the same runtime/execution-provider isolates the
quantization effect from unrelated backend differences (e.g. PyTorch-MPS vs ONNX-CPU).
"""

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ONNX_DIR = ROOT / "onnx"
REPORTS_DIR = ROOT / "reports"

ARCHS = ["baseline_cnn", "mobilenet_transfer"]
N_WARMUP = 10
N_RUNS = 200


def benchmark(onnx_path: Path) -> dict:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, 64, 64).astype(np.float32)

    for _ in range(N_WARMUP):
        sess.run(None, {input_name: x})

    latencies_ms = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        sess.run(None, {input_name: x})
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": statistics.mean(latencies_ms),
        "std_ms": statistics.stdev(latencies_ms),
        "p50_ms": statistics.median(latencies_ms),
        "p95_ms": sorted(latencies_ms)[int(0.95 * N_RUNS)],
        "size_kb": onnx_path.stat().st_size / 1024,
    }


def main():
    results = {}
    for arch_name in ARCHS:
        fp32_stats = benchmark(ONNX_DIR / f"{arch_name}_fp32.onnx")
        int8_stats = benchmark(ONNX_DIR / f"{arch_name}_int8.onnx")
        results[arch_name] = {"fp32": fp32_stats, "int8": int8_stats}
        speedup = fp32_stats["mean_ms"] / int8_stats["mean_ms"]
        shrink = fp32_stats["size_kb"] / int8_stats["size_kb"]
        print(f"{arch_name}:")
        print(f"  FP32: {fp32_stats['mean_ms']:.3f}ms +/- {fp32_stats['std_ms']:.3f}ms, {fp32_stats['size_kb']:.1f} KB")
        print(f"  INT8: {int8_stats['mean_ms']:.3f}ms +/- {int8_stats['std_ms']:.3f}ms, {int8_stats['size_kb']:.1f} KB")
        print(f"  -> {speedup:.2f}x latency, {shrink:.2f}x smaller")

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "latency_benchmark.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
