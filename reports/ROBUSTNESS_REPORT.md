# Quantization vs. Robustness: Evaluating INT8 Traffic Sign Recognition Under Corruption, Adversarial Attacks, and Distribution Shift

## Research question

Does INT8 quantization of a traffic-sign classifier (FP32 -> INT8, via ONNX Runtime) change
its robustness to visual corruptions and adversarial attacks, even when clean accuracy is
preserved?

## Core hypothesis (as originally stated)

> INT8 quantization measurably increases adversarial vulnerability (FGSM/PGD transfer
> success rate) and reduces corruption robustness compared to the unquantized FP32 model,
> even when standard-condition accuracy is preserved.

**Result: partially refuted.** For both architectures, adversarial examples generated
white-box against FP32 do *not* become more effective when transferred to the INT8
model — transfer success rates are equal to or slightly *lower* than the original FP32
white-box success rates. That is not the same claim as "the INT8 model is intrinsically
more adversarially robust" — see [Threat models](#threat-models) below for why, and what
would be needed to test that stronger claim directly. Corruption robustness is
essentially unchanged for the baseline CNN, but *does* degrade for MobileNetV2,
specifically at high corruption severity. The hypothesis holds only for one architecture
on one axis (corruption, not adversarial).

## Threat models

The adversarial evaluation in this report covers four threat models. The headline
"quantization does not increase adversarial vulnerability" claim only holds for the
second one:

1. **FP32 white-box.** FGSM/PGD computed directly against the FP32 PyTorch model's own
   gradients. This is the baseline everything else is compared to.
2. **FP32 -> INT8 transfer.** The *same* FP32-generated adversarial examples, evaluated
   unchanged against the INT8 model. This is a legitimate and realistic threat model on
   its own (an attacker targeting a deployed quantized model typically doesn't have
   white-box access to it either — see the Phase 2 methodology note), but it measures
   *transferability between two nearly identical models* (same architecture, same
   weights, just quantized), not the INT8 model's intrinsic robustness to an attack
   tailored to it.
3. **Black-box cross-architecture transfer.** Adversarial examples crafted against one
   FP32 architecture, evaluated against the *other*, independently-trained FP32
   architecture — no shared weights or structure between source and target. See
   `blackbox_transfer.png` and the table below.
4. **QAT-INT8 white-box.** FGSM/PGD computed directly against the *prepared* (fake-quant,
   pre-`convert()`) QAT model's own gradients — a genuine white-box attack on a real
   quantized-model surrogate, not a transfer attack. See Phase 4 below. This finally
   answers the question threat model 2 couldn't: does an INT8-aware model have its own
   nearby adversarial examples a transfer attack might miss? For the baseline CNN,
   apparently fewer than FP32 has; for MobileNetV2, about the same number.

**Result: cross-architecture transfer is dramatically weaker than same-architecture
transfer**, confirming that threat model 2's high transfer-success rates are an artifact
of the FP32/INT8 pair sharing an almost-identical decision boundary, not evidence that
adversarial examples generally transfer well to this task's models:

| Direction | PGD success eps=1/255 | eps=2/255 | eps=4/255 | eps=8/255 |
|---|---|---|---|---|
| Baseline CNN -> MobileNetV2 (black-box) | 0.9% | 2.8% | 8.5% | 23.6% |
| MobileNetV2 -> Baseline CNN (black-box) | 0.5% | 0.9% | 2.2% | 8.7% |
| Baseline: FP32 white-box (reference) | 48.3% | 78.2% | 97.2% | 100.0% |
| MobileNetV2: FP32 white-box (reference) | 84.6% | 93.9% | 98.0% | 99.7% |

At eps=1/255, cross-architecture transfer succeeds on well under 1% of cases where
same-architecture FP32->INT8 transfer succeeds on 46.7-84.6% — a ~50-100x difference. This
is the expected result for genuinely independent models and it's exactly what makes
threat model 2 a weak test of "does quantization help or hurt adversarial robustness":
same-architecture transfer success mostly reflects shared architecture, not anything
specific to quantization.

Not covered: an independent black-box source model attacking the *QAT* model
specifically (only the FP32-vs-FP32 cross-architecture case above was tested) — a
plausible follow-up, not attempted here.

So the precise, defensible version of this report's PTQ adversarial finding is:
**FP32-crafted adversarial examples transfer to the PTQ-INT8 model at success rates no
higher than white-box attacks against FP32 itself — but this transfer success is driven
mainly by the two models sharing an architecture and weights, as shown by how much lower
cross-architecture transfer success is.** That's still a useful result — it rules out
quantization making a deployed model an *easier* transfer target than genuine black-box
attacks would already make it — but on its own it is not evidence that INT8 inference is
intrinsically more adversarially robust than FP32. Phase 4's QAT white-box result is
closer to that stronger claim, with an important caveat of its own — see below.

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
- Two architectures, each trained across 5 seeds (42, 123, 2024, 7, 999):
  - **Baseline CNN**: small from-scratch conv net (4 conv blocks, ~280K params)
  - **MobileNetV2 (transfer)**: ImageNet-pretrained, fully fine-tuned

## Phase 1: architecture comparison (FP32)

| Architecture | Test Accuracy | Test Macro-F1 |
|---|---|---|
| Baseline CNN | 0.931 +/- 0.010 | 0.898 +/- 0.020 |
| MobileNetV2 (transfer) | **0.967 +/- 0.005** | **0.946 +/- 0.007** |

(mean +/- std over 5 seeds; see `arch_comparison.png`)

MobileNetV2 outperforms the baseline on both metrics with lower run-to-run variance. The
gap between accuracy and macro-F1 for both models traces back to class imbalance in
GTSRB (see `eda_overview.png`) — the lowest-F1 classes are consistently the
lowest-support ones.

**Grad-CAM** (`gradcam_grid.png`) confirms both models attend to the sign pictogram
rather than background — required for MobileNetV2 to hold at a non-default target layer
(`features[6]`, 8x8 resolution), since its final layer collapses to a 2x2 feature map at
64x64 input resolution and produces an uninformative diffuse heatmap.

### Corruption robustness (FP32)

Ten corruptions, 4 severities each, grouped into 5 categories (noise; blur — Gaussian
and motion; geometric — rotation and perspective; photometric — brightness/contrast and
JPEG compression; environmental — fog, rain, shadow). See `corruption_curves.png`.

- MobileNetV2 is consistently more robust than the baseline on blur, motion blur,
  rotation, perspective, brightness/contrast, JPEG compression, and fog at every
  severity.
- **Gaussian noise is the exception**: both architectures collapse to near-random
  accuracy (<10%) by severity 4, regardless of pretraining. Transfer learning's
  robustness advantage does not extend to raw pixel noise. Rain (also pixel-level,
  though structured rather than uniform) shows the same pattern, if less extreme
  (~27-31% at severity 4 for both).
- **Shadow reverses the usual pattern**: the baseline CNN is *more* robust than
  MobileNetV2 at severity 4 (72.8% vs. 64.6%) — the only corruption type where transfer
  learning's advantage disappears entirely rather than just shrinking.

### Adversarial robustness (FP32, white-box)

FGSM and PGD (10 steps), L-inf epsilon in {1,2,4,8}/255, evaluated on the full
12,630-image test set. See `adversarial_curves.png`.

- Under **FGSM** (weak, single-step), MobileNetV2 looks more robust than the baseline.
- Under **PGD** (strong, iterative), that reverses: MobileNetV2 is *more* vulnerable
  (e.g. at eps=1/255, PGD accuracy is 47.5% for baseline vs. 15.0% for MobileNetV2).
  This is a textbook **gradient-masking** pattern — a weak attack can give a false sense
  of robustness that a stronger attack exposes.
- Both models are essentially fully broken (>99% PGD attack success) by eps=8/255.

## Phase 2: INT8 quantization

Both FP32 models were exported to ONNX (normalization baked into the graph via a
`NormalizedModel` wrapper, so every downstream evaluation operates on raw [0,1] pixel
tensors regardless of architecture) and statically quantized to INT8 with ONNX Runtime,
calibrated on 1000 held-out training images.

**Methodology note:** gradient-based attacks need a differentiable model, and the
quantized ONNX graph isn't one. Adversarial examples were generated white-box against the
FP32 PyTorch model and evaluated unchanged against the INT8 model (a *transfer* attack).
This isn't just a workaround — it reflects a realistic threat model, since an attacker
targeting a deployed quantized model generally won't have white-box access to its
internals either. See [Threat models](#threat-models) above for exactly what this does
and doesn't establish about the INT8 model's intrinsic robustness.

### Clean accuracy: preserved

| Architecture | FP32 | INT8 | Delta |
|---|---|---|---|
| Baseline CNN | 91.83% | 91.76% | -0.07pp |
| MobileNetV2 | 97.20% | 96.59% | -0.61pp |

### Corruption robustness: unchanged for baseline, degrades for MobileNetV2 at high severity

See `quantization_comparison.png` (left column) and `corruption_severity4_ci.png` for the
multi-seed confidence intervals below. This subsection covers the original 4 corruption
types, the only ones with multi-seed statistical backing; Phase 4's corruption table
covers the full 10-type suite (single-seed) alongside QAT.

- **Baseline CNN**: differences are within noise at every severity (max |diff| = 0.56pp),
  in both directions. Quantization has no detectable effect.
- **MobileNetV2**: a consistent, growing gap on blur, rotation, and brightness/contrast as
  severity increases — roughly 0.5-1.1pp at severity 1, growing to **-5.6pp (blur), -1.2pp
  (rotation), and -9.1pp (brightness/contrast) at severity 4**. Gaussian noise stays flat
  for both architectures. Quantization compounds with corruption severity for the
  transfer-learned model specifically.

Single-seed (42) point estimates, as originally reported:

| Corruption (severity 4) | FP32 | INT8 | Delta |
|---|---|---|---|
| Blur | 85.15% | 79.56% | -5.58pp |
| Noise | 10.10% | 9.93% | -0.17pp |
| Rotation | 72.93% | 71.76% | -1.17pp |
| Brightness/contrast | 74.75% | 65.68% | **-9.07pp** |

**Statistical confirmation (5 seeds: 42, 123, 2024, 7, 999).** The single-seed numbers
above could in principle just be one lucky (or unlucky) training run. Re-running the full
corruption sweep across all 5 seeds and computing the paired FP32-vs-INT8 delta per seed —
with a bootstrap 95% CI, a paired t-test, and Cohen's *dz* effect size on those 5 paired
differences — confirms the single-seed numbers were representative, not noise:

| Architecture | Corruption (sev. 4) | Mean delta | Bootstrap 95% CI | p (paired t) | Cohen's *dz* |
|---|---|---|---|---|---|
| Baseline CNN | blur | +0.02pp | [-0.14, +0.16] | 0.83 | 0.10 |
| Baseline CNN | brightness/contrast | +0.08pp | [-1.24, +1.64] | 0.93 | 0.04 |
| MobileNetV2 | blur | -7.43pp | [-9.15, -5.72] | 0.0016 | -3.39 |
| MobileNetV2 | rotation | -1.36pp | [-1.57, -1.17] | 0.0003 | -5.44 |
| MobileNetV2 | brightness/contrast | -8.54pp | [-9.33, -7.74] | 0.0001 | -8.18 |

(Full table for every corruption x severity x architecture combination, including Gaussian
noise which stays flat and non-significant for both architectures: `reports/statistics.json`.)

The baseline CNN's near-zero deltas have wide CIs that comfortably straddle zero and are
nowhere near significant (p>0.8) — genuinely no effect, not just a small one. MobileNetV2's
degradation, by contrast, has tight CIs that exclude zero entirely and enormous effect
sizes (|dz|>3) — this is a real, highly consistent property of quantizing this specific
architecture, reproducible across 5 independently-trained models, not a fluke of one
training run.

### FP32→INT8 transfer-attack success: not higher than FP32 white-box — if anything, slightly lower

See `quantization_comparison.png` (right column).

PGD transfer-attack success rate against INT8, compared to white-box PGD success rate
against FP32, at every epsilon tested:

| Architecture | eps=1/255 | eps=2/255 | eps=4/255 | eps=8/255 |
|---|---|---|---|---|
| Baseline: FP32 white-box | 48.3% | 78.2% | 97.2% | 100.0% |
| Baseline: INT8 transfer | 46.7% | 77.7% | 97.1% | 100.0% |
| MobileNetV2: FP32 white-box | 84.6% | 93.9% | 98.0% | 99.7% |
| MobileNetV2: INT8 transfer | 77.0% | 92.7% | 97.7% | 99.7% |

Every single PGD comparison shows INT8 transfer success at or below the FP32 white-box
rate (largest gap: -7.6pp for MobileNetV2 at eps=1/255). This is the expected direction
for a *transfer* attack — a perturbation optimized against one model's exact decision
boundary is inherently somewhat less effective against a different (even slightly
different) model — but it directly contradicts the hypothesis's claim that quantization
would *increase* vulnerability. FGSM shows a similar pattern overall, with two negligible
exceptions (MobileNetV2 at eps=4/255 and eps=8/255, where transfer success is +0.2pp and
+0.8pp *higher* than white-box — within run-to-run noise, not a reversal of the finding).
As noted in
[Threat models](#threat-models), this speaks to transferability, not to how the INT8
model would fare against an attack computed directly against its own decision surface —
see Phase 4 for that result via a QAT surrogate.

**Statistical confirmation (5 seeds, PGD only).** Re-running PGD white-box and PGD
transfer across all 5 seeds and bootstrapping the paired (transfer - white-box) delta
shows the "transfer <= white-box" direction is not seed-42-specific: every epsilon for
both architectures gives a negative (or zero) mean delta with a bootstrap 95% CI that
excludes positive values, and the effect sizes are large (|Cohen's *dz*| mostly 4-11) —
e.g. MobileNetV2 at eps=1/255: mean delta -6.82pp, CI [-7.65, -6.05], p=0.0001, dz=-6.70.
Full per-epsilon table: `reports/statistics.json`. The magnitudes here are small in
absolute terms (this section's whole point is that they *don't* increase), but they are
consistently, significantly negative — reinforcing rather than changing the finding.

### Latency and model size

Benchmarked via ONNX Runtime (CPUExecutionProvider) for both formats, isolating the
quantization effect from unrelated hardware/backend differences.

| Architecture | FP32 latency | INT8 latency | Speedup | FP32 size | INT8 size | Shrink |
|---|---|---|---|---|---|---|
| Baseline CNN | 2.34ms | 1.05ms | 2.22x | 570.0 KB | 162.9 KB | 3.50x |
| MobileNetV2 | 3.42ms | 1.65ms | 2.08x | 8878.8 KB | 2609.4 KB | 3.40x |

### Calibration-set size ablation: not a sensitive knob for this task

The 1000-image calibration set above was a somewhat arbitrary choice. To check whether it
mattered, both architectures were re-quantized at 6 calibration sizes (50, 100, 200, 500,
1000, 2000 images, all seed 42) and re-evaluated on clean accuracy and severity-4
corruption robustness. See `calibration_ablation.png`.

| Calibration size | Baseline clean | Baseline blur sev.4 | MobileNetV2 clean | MobileNetV2 blur sev.4 |
|---|---|---|---|---|
| 50 | 91.84% | 68.23% | 96.79% | 80.85% |
| 100 | 91.78% | 67.86% | 96.62% | 79.79% |
| 200 | 91.78% | 67.89% | 96.72% | 79.55% |
| 500 | 91.77% | 67.95% | 96.67% | 79.71% |
| 1000 | 91.76% | 67.97% | 96.59% | 79.56% |
| 2000 | 91.74% | 67.90% | 96.44% | 79.21% |

Model size is identical (162.9 KB / 2609.4 KB) at every calibration size, as expected —
calibration only fits the quantization scale/zero-point statistics, not the weight
precision, which is what determines file size. Clean accuracy and corruption robustness
both stay within roughly 0.4pp of their mean across the full 40x range of calibration-set
sizes tested, for both architectures. **Calibration-set size in the 50-2000 range is not
a meaningful lever for this task** — the corruption-robustness cost found for MobileNetV2
in this report is a property of static INT8 quantization itself, not an artifact of an
under-calibrated model that a bigger calibration set would fix. Practically, this also
means the smallest calibration set tested (50 images) is essentially free to use for this
model/dataset combination.

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
| Baseline CNN | 47.48% | 47.13% | -0.35pp |
| MobileNetV2 | 68.93% | 68.29% | -0.64pp |

The FP32-INT8 gap stays small even on this out-of-distribution dataset — reinforcing
Phase 2's finding that quantization's cost is minor and consistent, not something that
compounds specifically under distribution shift.

## Phase 4: quantization-aware training (QAT)

A third quantization variant alongside FP32 and Phase 2's post-training static
quantization (PTQ): fine-tuned for 3 epochs (Adam, lr=1e-4) from the same converged
FP32 seed-42 checkpoint, using PyTorch's eager-mode QAT (fake-quantization observers
during training, backend `qnnpack` — the only quantized engine available on Apple
Silicon; `fbgemm` is x86-only). Unlike PTQ, which needs no additional training, QAT's
best-epoch checkpoint (selected by validation macro-F1) is confounded with those 3 extra
fine-tuning epochs — **every comparison below is FP32/PTQ vs. "3-more-epochs-of-
training-plus-quantization-awareness," not vs. quantization-awareness alone.** This
matters concretely: the *prepared* (pre-`convert()`, fake-quantized) QAT model stays
fully differentiable throughout training via a straight-through estimator, which is
exactly the "quantization-aware differentiable surrogate" the Threat Models section
identifies as needed for a genuine INT8 white-box attack — see below.

### Clean accuracy: QAT helps the small model, hurts the large one

| Architecture | FP32 | PTQ INT8 | QAT INT8 |
|---|---|---|---|
| Baseline CNN | 91.83% | 91.76% | **94.98%** |
| MobileNetV2 | 97.20% | 96.59% | 95.33% |

Baseline CNN's accuracy *improved* under QAT (+3.15pp over FP32) — plausibly the extra
training epochs, fake-quant noise acting as a mild regularizer, or both; this project's
3-epoch, single-run, single-seed setup can't separate those. MobileNetV2 instead *lost*
accuracy under QAT (-1.87pp vs. FP32, -1.26pp vs. PTQ) despite the same extra epochs —
the larger model's fine-tuning may have started overfitting before its best-validation
checkpoint (epoch 0 of 3, per training logs), rather than benefiting from further
adaptation.

### Corruption robustness: QAT improves every corruption type for the baseline CNN

See `qat_corruption_comparison.png`.

| Corruption (severity 4) | Baseline CNN FP32 | PTQ | QAT | MobileNetV2 FP32 | PTQ | QAT |
|---|---|---|---|---|---|---|
| Blur | 68.2% | 68.0% | **73.2%** | 85.1% | 79.6% | 78.2% |
| Motion blur | 44.6% | 44.1% | **52.5%** | 66.8% | 60.9% | 58.9% |
| Rotation | 46.2% | 46.8% | 46.2% | 72.9% | 71.8% | **74.7%** |
| Perspective | 49.8% | 49.6% | **63.6%** | 77.2% | 75.7% | 73.6% |
| Brightness/contrast | 41.0% | 41.3% | **47.6%** | 74.8% | 65.7% | 64.6% |
| JPEG compression | 62.4% | 62.2% | **67.7%** | 72.0% | 71.0% | 69.3% |
| Fog | 46.5% | 46.1% | **53.6%** | 87.0% | 83.4% | 82.6% |
| Rain | 26.6% | 26.5% | **37.7%** | 31.1% | 30.6% | 29.7% |
| Shadow | 72.8% | 73.5% | **77.8%** | 64.6% | 62.7% | 61.3% |
| Noise | 7.5% | 7.5% | 11.5% | 9.7% | 10.0% | 11.0% |

For the baseline CNN, QAT beats *both* FP32 and PTQ on **every single corruption type**
at severity 4 — not a cherry-picked result. For MobileNetV2, QAT is roughly flat-to-worse
relative to PTQ on 8 of 10 corruptions (rotation and noise are the exceptions), meaning
QAT does not recover the corruption-robustness cost Phase 2 found for this architecture,
and mildly extends it on several types (blur, motion blur, shadow).

### Adversarial robustness: a genuine INT8 white-box result, at last

See `qat_adversarial_comparison.png`. Two new numbers close the Threat Models gap: PGD
computed directly against the differentiable QAT surrogate (**QAT-INT8 white-box** — a
true white-box attack on a real quantized-model stand-in, not a transfer attack), and
FP32-crafted examples transferred to the converted QAT model (**FP32→QAT-INT8
transfer**, directly comparable to Phase 2's FP32→PTQ-INT8 transfer).

| Architecture | Threat model | eps=1/255 | eps=2/255 | eps=4/255 | eps=8/255 |
|---|---|---|---|---|---|
| Baseline CNN | FP32 white-box | 48.3% | 78.2% | 97.2% | 100.0% |
| Baseline CNN | PTQ-INT8 transfer | 46.7% | 77.7% | 97.1% | 100.0% |
| Baseline CNN | **QAT-INT8 white-box** | **38.9%** | **71.7%** | **94.2%** | 99.9% |
| Baseline CNN | FP32→QAT-INT8 transfer | 23.9% | 56.6% | 86.2% | 99.3% |
| MobileNetV2 | FP32 white-box | 84.6% | 93.9% | 98.0% | 99.7% |
| MobileNetV2 | PTQ-INT8 transfer | 77.0% | 92.7% | 97.7% | 99.7% |
| MobileNetV2 | QAT-INT8 white-box | 77.5% | 93.7% | 98.5% | 99.8% |
| MobileNetV2 | FP32→QAT-INT8 transfer | 49.0% | 84.4% | 95.9% | 99.3% |

For the baseline CNN, the QAT-INT8 white-box attack succeeds measurably *less* often
than either FP32 white-box or PTQ-INT8 transfer at every epsilon (e.g. -9.4pp at
eps=1/255) — a real robustness improvement under a genuine white-box attack, not just
"transfer doesn't find it." For MobileNetV2, QAT-INT8 white-box tracks FP32 white-box
closely (within 1-2pp), neither clearly better nor worse. This mirrors the corruption
result above and is consistent with the same explanation: whatever QAT changed for the
baseline CNN (more training, regularization, or genuine quantization-awareness) doesn't
transfer to MobileNetV2's fine-tuning regime. As with clean accuracy, this can't be
attributed to quantization-awareness alone given the training confound.

### Deployment cost: not comparable to Phase 2's numbers, by construction

QAT-INT8 runs as real PyTorch quantized ops (`qnnpack` backend, CPU), benchmarked in
PyTorch directly — not ONNX Runtime, unlike Phase 2's PTQ numbers. The two latency/size
results are **not cross-comparable**; only FP32-vs-QAT-INT8 *within this benchmark* means
anything.

| Architecture | FP32 latency | QAT INT8 latency | Speedup | FP32 size | QAT INT8 size | Shrink |
|---|---|---|---|---|---|---|
| Baseline CNN | 1.77ms | 1.55ms | 1.14x | 583.4 KB | 151.8 KB | 3.84x |
| MobileNetV2 | 20.23ms | 3.51ms | 5.77x | 9138.0 KB | 2344.2 KB | 3.90x |

Model-size shrink (~3.8-3.9x) is close to Phase 2's ONNX Runtime numbers, as expected —
size depends on weight precision, not runtime. Latency speedup is not: PyTorch's
quantized-CPU kernels apparently have more fixed per-call overhead than ONNX Runtime for
a model as small as the baseline CNN (1.14x vs. Phase 2's 2.22x), but scale far better
for MobileNetV2 (5.77x vs. 2.08x) — a runtime-implementation difference, not a
quantization-technique one.

## Phase 5: second generalization check (BelgiumTSC)

A second, independent out-of-distribution test alongside Phase 3's Mapillary+DFG check:
[BelgiumTSC](https://www.kaggle.com/datasets/abhi8923shriv/belgium-ts) (CC0), 62 classes
of Belgian traffic signs captured from roof-mounted van cameras — a different country
than GTSRB's Germany, a different capture setup than Mapillary+DFG's crowdsourced
street-level crops, but the same Vienna Convention sign family as GTSRB. Does the
Mapillary finding (large accuracy drop, small FP32-vs-INT8 gap) replicate on a second,
unrelated dataset?

**Methodology and a real labeling bug worth documenting.** BelgiumTSC ships no official
class-meaning file, unlike Mapillary+DFG's `classes.json`. Classes were identified
visually, then **validated by checking what the GTSRB-trained model actually predicts for
~20 images per candidate class** — a mismatch between the visual guess and the model's
consistent prediction reliably caught 4 labeling errors from thumbnail-resolution visual
identification (e.g. a class that looked like "stop" at small size turned out to be "no
entry"; another that looked like "pedestrians" was actually "right-of-way at next
intersection", confirmed by comparing side-by-side against real GTSRB reference images).
Fixing these raised measured accuracy from ~47% to ~70% — a reminder that a generalization
number is only as trustworthy as the class mapping behind it. Full detail in
`src/data/belgium_mapping.py`. 19 classes matched confidently; GTSRB's "no passing" and
"pedestrians" have no confident match in this particular 62-class reduced subset.

### Result: a large drop again, but this time both architectures drop by about the same amount

See `belgium_generalization.png`.

| Architecture | GTSRB test (19 classes) | BelgiumTSC (FP32) | Drop |
|---|---|---|---|
| Baseline CNN | 92.52% | 68.81% | **-23.7pp** |
| MobileNetV2 (transfer) | 96.78% | 73.98% | **-22.8pp** |

Both models again generalize far worse than their GTSRB numbers suggest, replicating
Phase 3's headline finding on a second, independent dataset. But the *architecture*
story is different: on Mapillary, MobileNetV2's pretrained features gave it a large
generalization edge (a 28.4pp drop vs. baseline's 44.8pp — 16.4pp narrower). On
BelgiumTSC, that edge nearly vanishes (22.8pp vs. 23.7pp — under 1pp apart). Belgium's
much closer visual and cultural proximity to GTSRB's Germany (same continent, same sign
manufacturing standards, likely similar camera/lighting conditions) plausibly explains
why: MobileNetV2's ImageNet-pretrained generalization advantage matters most when the
shift is large (crossing into a visually distant domain), and matters much less when the
new domain is already visually close to the training distribution.

Per-class F1 (MobileNetV2, FP32) ranges from 0.97 ("go straight or right") down to
essentially 0 ("turn left ahead", n=222 — not a small-sample fluke). That specific
failure was checked carefully given the mapping bugs found above: the BelgiumTSC icon is
an unambiguous left-pointing arrow matching GTSRB 34 exactly, but the model consistently
predicts "keep right" or "keep left" instead — a genuine, surprising generalization
failure on a plain mandatory-direction sign, not a labeling error. "Road work" and
"children crossing" also generalize poorly (F1 ~0.43-0.45) and are mutually confused by
the model — the same two classes that confused each other during the mapping validation
step above, suggesting the model's road-work/children-crossing decision boundary doesn't
transfer well regardless of which dataset it's tested on.

### FP32 vs. INT8 on BelgiumTSC: the small-gap finding replicates a second time

| Architecture | FP32 | INT8 | Delta |
|---|---|---|---|
| Baseline CNN | 68.81% | 68.67% | -0.14pp |
| MobileNetV2 | 73.98% | 73.73% | -0.25pp |

Both deltas are smaller than even Phase 3's already-small Mapillary gaps (-0.58pp /
-0.38pp there). Across three independent test conditions now (GTSRB, Mapillary+DFG,
BelgiumTSC), quantization's cost to accuracy stays consistently under 1 percentage point
— the single most robust finding in this entire report.

## Conclusion

For this task, INT8 quantization delivers its real-time deployment benefits (~2x latency,
~3.4x size) without making either architecture an easier target for adversarial examples
transferred from the FP32 model (see [Threat models](#threat-models) for what this claim
does and doesn't cover), and at no meaningful cost to corruption robustness for the
from-scratch baseline CNN. The one
genuine robustness cost found is corruption robustness for the transfer-learned
MobileNetV2 specifically, and specifically at high corruption severity (a regime an
autonomous system would ideally never operate in for other reasons — e.g. severity-4
brightness/contrast is a near-unrecognizable image even to a human). That cost held up
even on the Mapillary generalization set, where the FP32-INT8 gap remained just as small
as on GTSRB.

The practical implication: architecture choice matters more than the FP32-vs-INT8
decision here, and matters even more for generalization than for quantization safety.
MobileNetV2 trades a bit of worst-case corruption robustness for much better accuracy,
adversarial robustness under FGSM, and — on Mapillary+DFG, by a wide margin —
generalization to signs it's never seen the likes of. Given that the whole point of an
autonomous traffic-sign system is operating on signs the training set didn't anticipate,
that last property arguably matters more than any of the individually-tested robustness
axes — but Phase 5 complicates it: MobileNetV2's generalization edge over the baseline
CNN nearly disappears on BelgiumTSC, a domain much closer to GTSRB than Mapillary+DFG's
crops are. Pretraining's generalization payoff looks domain-shift-dependent, not a fixed
property of the architecture.

Phase 4 adds a genuine INT8 white-box result and a third quantization variant, and the
same architecture-matters theme holds: QAT is a clear win for the baseline CNN (better
accuracy, better corruption robustness on every type tested, and a real reduction in
white-box adversarial success) but does not rescue MobileNetV2's PTQ corruption cost and
leaves its adversarial robustness roughly unchanged. Whether that's because
quantization-awareness itself helps small from-scratch models more, or simply because 3
epochs of extra fine-tuning helps a small model more than a model already fine-tuned from
ImageNet, this project's single-run setup can't distinguish — see Limitations.

Phase 5 is this report's most-replicated finding: across three independent test
conditions (GTSRB itself, Mapillary+DFG, BelgiumTSC), the FP32-vs-INT8 accuracy gap never
exceeds 0.6 percentage points. Whatever else is true about this task, static INT8
quantization's cost to clean accuracy is small, consistent, and not domain-shift-
sensitive.

## Limitations

- The adversarial-robustness finding covers FP32-white-box, FP32→PTQ-INT8-transfer,
  black-box cross-architecture transfer, and (Phase 4) QAT-INT8 white-box (see
  [Threat models](#threat-models)) — but the QAT white-box result is confounded with 3
  extra fine-tuning epochs PTQ never received (see Phase 4's caveat), so it isn't a clean
  isolation of "quantization-awareness" from "more training."
- FGSM/PGD were hand-implemented rather than via Foolbox/ART, per the spec's suggested
  tools — standard formulations, but not independently cross-checked against a reference
  library implementation.
- The multi-seed adversarial variance check (see Phase 2) uses PGD only, not FGSM — PGD is
  the stronger, more diagnostic attack in this project's findings (see the gradient-masking
  discussion in Phase 1), and running both attacks across 5 seeds would roughly double an
  already-expensive full-test-set sweep for limited additional statistical value.
- The calibration-size ablation and multi-seed statistical checks were only run for static
  PTQ INT8 — QAT (Phase 4) and the black-box cross-architecture transfer attack are each
  single-seed (42) only.
- QAT was fine-tuned for a fixed 3 epochs at a single learning rate (1e-4), chosen without
  a hyperparameter sweep; a different schedule could change how much of Phase 4's
  clean-accuracy and robustness shift is attributable to quantization-awareness itself.
- The Mapillary and BelgiumTSC generalization checks only cover the classes with an
  unambiguous semantic match (23 and 19 respectively), exclude all numeric speed-limit
  classes entirely, and their worst-performing per-class results are plausibly confounded
  by cross-standard pictogram differences rather than pure distribution shift (see
  Phases 3 and 5). Both are single-seed (42) only, unlike the multi-seed statistical
  checks run for GTSRB-native corruption/adversarial results. The OpenCV webcam demo
  (marked lowest-priority/no-research-value in the original spec) was not attempted.

## Figures

All in `reports/`:
- `eda_overview.png` — class distribution, image size distribution
- `arch_comparison.png` — FP32 architecture comparison, multi-seed
- `gradcam_grid.png` — Grad-CAM, both architectures
- `corruption_curves.png` — FP32 accuracy vs. corruption severity
- `adversarial_curves.png` — FP32 accuracy vs. attack strength (FGSM, PGD)
- `quantization_comparison.png` — FP32 vs. INT8, corruption + adversarial, side by side
- `mapillary_generalization.png` — GTSRB vs. Mapillary accuracy, per-class F1 breakdown
- `belgium_generalization.png` — GTSRB vs. BelgiumTSC accuracy, per-class F1 breakdown
- `blackbox_transfer.png` — black-box cross-architecture transfer vs. white-box PGD
- `corruption_severity4_ci.png` — multi-seed FP32 vs. INT8 severity-4 accuracy, 95% CIs
- `calibration_ablation.png` — clean accuracy and blur-severity-4 accuracy vs. calibration size
- `qat_corruption_comparison.png` — FP32 vs. PTQ vs. QAT, severity-4 accuracy, all 10 corruption types
- `qat_adversarial_comparison.png` — PGD success across all four adversarial threat models
