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

## Phase 3: generalization check (Mapillary)

GTSRB is exclusively German, standardized signage, captured from sequential dashcam video
— a model could score well on it while having learned something closer to "recognize
this specific country's sign fonts and image conditions" than "recognize traffic signs."
This phase tests whether the trained models generalize to a different, more diverse
real-world image distribution: the [Mapillary+DFG dataset](https://www.kaggle.com/datasets/nomihsa965/traffic-signs-dataset-mapillary-and-dfg)
(pre-cropped signs from Africa-region street imagery, a different labeling taxonomy than
GTSRB's numeric classes).

**Methodology:** GTSRB's 43 classes were mapped to Mapillary's 76 classes by semantic
meaning (see `src/data/mapillary_mapping.py`); only unambiguous 1:1 matches were kept —
23 of 43 GTSRB classes. Notably excluded: all 9 numeric speed-limit classes, since
Mapillary lumps every speed value into one generic "maximum-speed-limit" class with no
way to recover which specific limit (20/30/.../120 km/h) a given crop shows. 150 images
per mapped class were sampled (3,450 total). For a fair comparison, GTSRB test accuracy
was recomputed restricted to the same 23 classes (rather than reusing the full-43-class
headline numbers), since the two accuracies would otherwise be measuring different tasks.

### Result: both models generalize far worse than their GTSRB numbers suggest

See `mapillary_generalization.png`.

| Architecture | GTSRB test (23 classes) | Mapillary (FP32) | Drop |
|---|---|---|---|
| Baseline CNN | 92.30% | 47.48% | **-44.8pp** |
| MobileNetV2 (transfer) | 97.38% | 68.93% | **-28.4pp** |

Both models lose enormous accuracy on the shifted distribution — confirming the concern
the project spec raised about GTSRB-specific results. MobileNetV2's ImageNet-pretrained
features generalize meaningfully better than the from-scratch CNN's (a 28pp drop vs. a
45pp drop), consistent with Phase 1's corruption-robustness finding that pretraining
helps against distribution shift generally, just not against pixel-level noise or (as
seen here) country-of-origin shift specifically.

Per-class F1 on Mapillary is highly uneven (MobileNetV2: 0.98 for "yield" down to 0.12
for "pedestrians"). The worst-performing classes are plausibly a mapping-methodology
artifact rather than a pure generalization failure — e.g. GTSRB's "pedestrians" warning
triangle and Mapillary's "pedestrians-crossing" sign may use visually different
pictograms across sign-design standards, which a semantic name match can't detect. This
is a genuine limitation of the class-mapping approach, not necessarily of the models.

### FP32 vs. INT8 on Mapillary: the earlier finding holds even under distribution shift

| Architecture | FP32 | INT8 | Delta |
|---|---|---|---|
| Baseline CNN | 47.48% | 46.90% | -0.58pp |
| MobileNetV2 | 68.93% | 68.55% | -0.38pp |

The FP32-INT8 gap stays small even on this out-of-distribution dataset — reinforcing
Phase 2's finding that quantization's cost is minor and consistent, not something that
compounds specifically under distribution shift.

## Conclusion

For this task, INT8 quantization delivers its real-time deployment benefits (~2x latency,
~3.4x size) at effectively no cost to adversarial robustness for either architecture, and
at no meaningful cost to corruption robustness for the from-scratch baseline CNN. The one
genuine robustness cost found is corruption robustness for the transfer-learned
MobileNetV2 specifically, and specifically at high corruption severity (a regime an
autonomous system would ideally never operate in for other reasons — e.g. severity-4
brightness/contrast is a near-unrecognizable image even to a human). That cost held up
even on the Mapillary generalization set, where the FP32-INT8 gap remained just as small
as on GTSRB.

The practical implication: architecture choice matters more than the FP32-vs-INT8
decision here, and matters even more for generalization than for quantization safety.
MobileNetV2 trades a bit of worst-case corruption robustness for much better accuracy,
adversarial robustness under FGSM, and — by a wide margin — generalization to signs it's
never seen the likes of. Given that the whole point of an autonomous traffic-sign system
is operating on signs the training set didn't anticipate, that last property arguably
matters more than any of the individually-tested robustness axes.

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
- The Mapillary generalization check only covers the 23 GTSRB classes with an
  unambiguous semantic match, excludes all numeric speed-limit classes entirely, and its
  worst-performing per-class results are plausibly confounded by cross-standard
  pictogram differences rather than pure distribution shift (see Phase 3). The OpenCV
  webcam demo (marked lowest-priority/no-research-value in the original spec) was not
  attempted.

## Figures

All in `reports/`:
- `eda_overview.png` — class distribution, image size distribution
- `arch_comparison.png` — FP32 architecture comparison, multi-seed
- `gradcam_grid.png` — Grad-CAM, both architectures
- `corruption_curves.png` — FP32 accuracy vs. corruption severity
- `adversarial_curves.png` — FP32 accuracy vs. attack strength (FGSM, PGD)
- `quantization_comparison.png` — FP32 vs. INT8, corruption + adversarial, side by side
- `mapillary_generalization.png` — GTSRB vs. Mapillary accuracy, per-class F1 breakdown
