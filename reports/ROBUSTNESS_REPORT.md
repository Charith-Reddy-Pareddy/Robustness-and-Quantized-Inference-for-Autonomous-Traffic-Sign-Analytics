# Robustness and Quantized Inference for Autonomous Traffic Sign Analytics

## Research question

Does INT8 quantization of a traffic-sign classifier (FP32 -> INT8, via ONNX Runtime) change
its robustness to visual corruptions and adversarial attacks, even when clean accuracy is
preserved?

## Core hypothesis (as originally stated)

> INT8 quantization measurably increases adversarial vulnerability (FGSM/PGD transfer
> success rate) and reduces corruption robustness compared to the unquantized FP32 model,
> even when standard-condition accuracy is preserved.

**Result: partially refuted.** Adversarial vulnerability does *not* increase under
quantization — for both architectures, transfer-attack success rates against the INT8
model are equal to or slightly *lower* than white-box success rates against FP32.
Corruption robustness is essentially unchanged for the baseline CNN, but *does* degrade
for MobileNetV2, specifically at high corruption severity. The hypothesis holds only for
one architecture on one axis (corruption, not adversarial).

## Dataset and methodology

- **GTSRB** (German Traffic Sign Recognition Benchmark), 43 classes, 39,209 training
  images, 12,630 held-out test images.
- GTSRB training images are sequential video frames of the same physical sign — a naive
  random split leaks near-duplicate frames across train/val. A **track-aware split**
  (grouped by (ClassId, TrackId), stratified by class) was used instead: 33,599 train /
  5,610 val, confirmed zero group overlap.
- No train-time augmentation (rotation, brightness, noise) was applied, since those exact
  perturbations are used as robustness *test* corruptions later — training on them would
  confound "robust because of architecture/quantization" with "robust because the model
  already saw this during training."
- Two architectures, each trained across 3 seeds (42, 123, 2024):
  - **Baseline CNN**: small from-scratch conv net (4 conv blocks, ~280K params)
  - **MobileNetV2 (transfer)**: ImageNet-pretrained, fully fine-tuned

## Phase 1: architecture comparison (FP32)

| Architecture | Test Accuracy | Test Macro-F1 |
|---|---|---|
| Baseline CNN | 0.926 +/- 0.008 | 0.887 +/- 0.019 |
| MobileNetV2 (transfer) | **0.968 +/- 0.004** | **0.947 +/- 0.006** |

(mean +/- std over 3 seeds; see `arch_comparison.png`)

MobileNetV2 outperforms the baseline on both metrics with lower run-to-run variance. The
gap between accuracy and macro-F1 for both models traces back to class imbalance in
GTSRB (see `eda_overview.png`) — the lowest-F1 classes are consistently the
lowest-support ones.

**Grad-CAM** (`gradcam_grid.png`) confirms both models attend to the sign pictogram
rather than background — required for MobileNetV2 to hold at a non-default target layer
(`features[6]`, 8x8 resolution), since its final layer collapses to a 2x2 feature map at
64x64 input resolution and produces an uninformative diffuse heatmap.

### Corruption robustness (FP32)

Four corruptions (blur, Gaussian noise, rotation, brightness/contrast shift), 4
severities each. See `corruption_curves.png`.

- MobileNetV2 is consistently more robust than the baseline on blur, rotation, and
  brightness/contrast at every severity.
- **Gaussian noise is the exception**: both architectures collapse to near-random
  accuracy (<10%) by severity 4, regardless of pretraining. Transfer learning's
  robustness advantage does not extend to raw pixel noise.

### Adversarial robustness (FP32, white-box)

FGSM and PGD (10 steps), L-inf epsilon in {1,2,4,8}/255, evaluated on a 2000-image
class-stratified subsample. See `adversarial_curves.png`.

- Under **FGSM** (weak, single-step), MobileNetV2 looks more robust than the baseline.
- Under **PGD** (strong, iterative), that reverses: MobileNetV2 is *more* vulnerable
  (e.g. at eps=1/255, PGD accuracy is 47.1% for baseline vs. 14.8% for MobileNetV2).
  This is a textbook **gradient-masking** pattern — a weak attack can give a false sense
  of robustness that a stronger attack exposes.
- Both models are essentially fully broken (>99% PGD attack success) by eps=8/255.

## Phase 2: INT8 quantization

Both FP32 models were exported to ONNX (normalization baked into the graph via a
`NormalizedModel` wrapper, so every downstream evaluation operates on raw [0,1] pixel
tensors regardless of architecture) and statically quantized to INT8 with ONNX Runtime,
calibrated on 200 held-out training images.

**Methodology note:** gradient-based attacks need a differentiable model, and the
quantized ONNX graph isn't one. Adversarial examples were generated white-box against the
FP32 PyTorch model and evaluated unchanged against the INT8 model (a *transfer* attack).
This isn't just a workaround — it reflects a realistic threat model, since an attacker
targeting a deployed quantized model generally won't have white-box access to its
internals either.

### Clean accuracy: preserved

| Architecture | FP32 | INT8 | Delta |
|---|---|---|---|
| Baseline CNN | 91.83% | 91.78% | -0.05pp |
| MobileNetV2 | 97.20% | 96.72% | -0.48pp |

### Corruption robustness: unchanged for baseline, degrades for MobileNetV2 at high severity

See `quantization_comparison.png` (left column).

- **Baseline CNN**: differences are within noise at every severity (max |diff| = 0.6pp),
  in both directions. Quantization has no detectable effect.
- **MobileNetV2**: a consistent, *monotonically growing* gap as severity increases —
  roughly 0.5-1pp at severity 1, growing to **-5.6pp (blur) and -9.2pp
  (brightness/contrast) at severity 4**. Quantization compounds with corruption severity
  for the transfer-learned model specifically.

| Corruption (severity 4) | FP32 | INT8 | Delta |
|---|---|---|---|
| Blur | 85.15% | 79.55% | -5.60pp |
| Noise | 10.10% | 9.85% | -0.25pp |
| Rotation | 72.93% | 72.13% | -0.80pp |
| Brightness/contrast | 74.75% | 65.58% | **-9.17pp** |

### Adversarial vulnerability: not increased — if anything, slightly reduced

See `quantization_comparison.png` (right column).

PGD transfer-attack success rate against INT8, compared to white-box PGD success rate
against FP32, at every epsilon tested:

| Architecture | eps=1/255 | eps=2/255 | eps=4/255 | eps=8/255 |
|---|---|---|---|---|
| Baseline: FP32 white-box | 48.4% | 79.5% | 97.0% | 100.0% |
| Baseline: INT8 transfer | 47.3% | 78.5% | 96.8% | 100.0% |
| MobileNetV2: FP32 white-box | 84.8% | 93.6% | 98.3% | 99.7% |
| MobileNetV2: INT8 transfer | 77.8% | 92.2% | 97.9% | 99.7% |

Every single comparison shows INT8 transfer success at or below the FP32 white-box rate
(largest gap: -7.0pp for MobileNetV2 at eps=1/255). This is the expected direction for a
*transfer* attack — a perturbation optimized against one model's exact decision boundary
is inherently somewhat less effective against a different (even slightly different)
model — but it directly contradicts the hypothesis's claim that quantization would
*increase* vulnerability. FGSM shows the same pattern.

### Latency and model size

Benchmarked via ONNX Runtime (CPUExecutionProvider) for both formats, isolating the
quantization effect from unrelated hardware/backend differences.

| Architecture | FP32 latency | INT8 latency | Speedup | FP32 size | INT8 size | Shrink |
|---|---|---|---|---|---|---|
| Baseline CNN | 2.34ms | 1.05ms | 2.22x | 570.0 KB | 162.9 KB | 3.50x |
| MobileNetV2 | 3.42ms | 1.65ms | 2.08x | 8878.8 KB | 2609.4 KB | 3.40x |

## Conclusion

For this task, INT8 quantization delivers its real-time deployment benefits (~2x latency,
~3.4x size) at effectively no cost to adversarial robustness for either architecture, and
at no meaningful cost to corruption robustness for the from-scratch baseline CNN. The one
genuine robustness cost found is corruption robustness for the transfer-learned
MobileNetV2 specifically, and specifically at high corruption severity (a regime an
autonomous system would ideally never operate in for other reasons — e.g. severity-4
brightness/contrast is a near-unrecognizable image even to a human).

The practical implication: architecture choice matters more than the FP32-vs-INT8
decision here. If deploying INT8, the from-scratch CNN's robustness profile survives
quantization essentially unchanged, while MobileNetV2 trades some worst-case corruption
robustness for its much higher baseline accuracy and robustness everywhere else.

## Limitations

- Adversarial evaluation used a 2000-image class-stratified subsample of the test set
  (not the full 12,630), since PGD's iterative forward+backward passes make a full sweep
  across 4 epsilons x 2 attacks x 2 models expensive. Corruption evaluation used the full
  test set.
- FGSM/PGD were hand-implemented rather than via Foolbox/ART, per the spec's suggested
  tools — standard formulations, but not independently cross-checked against a reference
  library implementation.
- Static INT8 quantization was calibrated on only 200 training images; a larger
  calibration set might shift the corruption-robustness gap found for MobileNetV2 in
  either direction.
- Mapillary generalization check and the OpenCV webcam demo (both marked optional/lowest
  priority in the original spec) were not attempted.

## Figures

All in `reports/`:
- `eda_overview.png` — class distribution, image size distribution
- `arch_comparison.png` — FP32 architecture comparison, multi-seed
- `gradcam_grid.png` — Grad-CAM, both architectures
- `corruption_curves.png` — FP32 accuracy vs. corruption severity
- `adversarial_curves.png` — FP32 accuracy vs. attack strength (FGSM, PGD)
- `quantization_comparison.png` — FP32 vs. INT8, corruption + adversarial, side by side
