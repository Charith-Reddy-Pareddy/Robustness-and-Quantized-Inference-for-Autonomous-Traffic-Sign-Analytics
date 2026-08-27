# Plan: Robustness and Quantized Inference for Autonomous Traffic Sign Analytics

Research question: does INT8 quantization of a traffic-sign classifier change its
robustness to corruptions and adversarial attacks, even when clean accuracy is preserved?

No git repo is created until the project is functionally complete. Once everything
works end to end, git gets initialized and the whole thing is pushed to GitHub in one go.
Work proceeds one day/session at a time.

- [x] Day 1 — Scaffold + data ingestion (GTSRB via Kaggle API, track-aware split, EDA)
- [x] Day 2 — Baseline CNN: train + evaluate (accuracy, macro-F1, confusion matrix, per-class)
      -> test acc 0.918, test macro-F1 0.861, seed 42, early-stopped epoch 10
- [x] Day 3 — Transfer-learning model (MobileNetV2) + multi-seed runs (2-3 seeds) for both archs
      -> baseline_cnn: acc 0.926+/-0.008, macro-F1 0.887+/-0.019 (seeds 42/123/2024)
      -> mobilenet_transfer: acc 0.968+/-0.004, macro-F1 0.947+/-0.006 (seeds 42/123/2024)
- [x] Day 4 — Corruption robustness (blur, noise, rotation, brightness/contrast) + Grad-CAM
      -> MobileNetV2 more robust than baseline on blur/rotation/brightness at every severity
      -> Gaussian noise: both architectures collapse almost identically (transfer learning
         gives no protection here) -- <10% accuracy at severity 4 for both
      -> Grad-CAM (features[6], 8x8 res -- final MobileNetV2 layer is too coarse at 64x64
         input) confirms both models attend to the sign pictogram, not background
- [x] Day 5 — Adversarial robustness: FGSM + PGD on FP32 models, attack success rate
      -> Hand-rolled FGSM/PGD (not Foolbox/ART) operating in raw pixel space via
         NormalizedModel wrapper, so epsilon is comparable across architectures
      -> Surprise finding: MobileNetV2 more robust to FGSM but MUCH more vulnerable to
         PGD than baseline (e.g. eps=1/255: baseline PGD acc 0.47 vs mobilenet 0.15) --
         classic gradient-masking pattern, FGSM alone would have given false confidence
      -> Both models -> ~0% accuracy / ~100% PGD attack success by eps=8/255
- [x] Day 6 — INT8 quantization (ONNX Runtime), re-test corruption + transfer-attacks on INT8,
      latency/size benchmark, robustness report
      -> Core hypothesis PARTIALLY REFUTED: adversarial vulnerability NOT increased by
         quantization (INT8 transfer success <= FP32 white-box success, every case);
         corruption robustness unchanged for baseline CNN, but degrades for MobileNetV2
         at high severity (-9.2pp brightness/contrast sev4, -5.6pp blur sev4)
      -> Latency ~2.1-2.2x faster, size ~3.4-3.5x smaller (INT8 vs FP32, ONNX Runtime CPU)
      -> Full report: reports/ROBUSTNESS_REPORT.md
- [x] Day 7 — Added regression test coverage for evaluate.summarize() and GTSRBDataset
      -> 6 new regression tests added covering macro-F1 calculation in
         evaluate.summarize() and ROI cropping in GTSRBDataset, which had no prior
         coverage. Full suite: 16 -> 22 tests, all passing.
- [x] Day 8 — git init + push finished project to GitHub in one commit history
      -> https://github.com/Charith-Reddy-Pareddy/Robustness-and-Quantized-Inference-for-Autonomous-Traffic-Sign-Analytics

- [x] Day 9 (extra) — Mapillary generalization check
      -> Mapped 23/43 GTSRB classes to the Mapillary+DFG dataset by semantic meaning
         (excluded: all 9 numeric speed-limit classes, no other clean matches)
      -> Both models generalize far worse than GTSRB numbers suggest: baseline_cnn
         92.3% (GTSRB, same 23 classes) -> 47.5% (Mapillary), -44.8pp; mobilenet_transfer
         97.4% -> 68.9%, -28.4pp. Transfer learning generalizes much better.
      -> FP32-vs-INT8 gap stays small even under this distribution shift (-0.6pp / -0.4pp)
         -- Day 6's quantization finding holds up out-of-distribution too
      -> Report updated: reports/ROBUSTNESS_REPORT.md Phase 3

Cut (lowest priority, no research value per spec): OpenCV webcam demo.

## Extensions (post-review)

Driven by external feedback on the finished project. Each item gets its own commit(s).

- [x] Sharpen title, add Threat models section distinguishing FP32 white-box from
      FP32->INT8 transfer attacks
- [x] Pin exact dependency versions, document test environment
- [x] Add experiment matrix table to README
- [x] Add CI (GitHub Actions, pytest on push/PR)
- [x] More seeds (7, 999 added to the original 3) and full-test-set adversarial eval
      (previously a 2000-image stratified subsample), higher INT8 calibration set (1000,
      up from 200)
      -> Core finding holds on the full test set and at the larger calibration size;
         corruption robustness degradation for MobileNetV2 became slightly clearer on
         rotation specifically (-1.17pp at severity 4, up from -0.80pp)
- [x] Black-box threat model: cross-architecture adversarial transfer (baseline_cnn <->
      mobilenet_transfer), no new training needed
      -> Confirms the FP32->INT8 transfer finding was mostly an artifact of shared
         architecture: cross-arch PGD transfer succeeds on <1% of cases at eps=1/255,
         vs. 46.7-84.6% for same-architecture FP32->INT8 transfer
- [x] Statistical rigor: mean +/- std / bootstrap CIs / significance tests for corruption
      and adversarial robustness deltas across seeds (currently only clean accuracy is
      multi-seed; robustness evals run on a single seed)
      -> Corruption robustness re-run across all 5 seeds (FP32 + INT8), PGD adversarial
         re-run across all 5 seeds (white-box + transfer). Baseline CNN's near-zero
         corruption deltas confirmed genuinely non-significant (p>0.8); MobileNetV2's
         degradation confirmed real and consistent (p<0.002, |Cohen's dz|>3 on blur/
         rotation/brightness-contrast at severity 4). See ROBUSTNESS_REPORT.md's new
         "Statistical confirmation" subsections and reports/statistics.json.
- [x] Calibration-set size ablation (50/100/200/500/1000/2000): accuracy, robustness
      delta, latency at each size
      -> Both architectures re-quantized at all 6 sizes. Model size is identical at
         every size (calibration doesn't affect weight precision); clean accuracy and
         severity-4 blur accuracy both stay within ~0.4pp of their mean across the full
         40x range. Calibration size is not a meaningful lever for this task -- the
         MobileNetV2 corruption-robustness cost is a property of static INT8
         quantization itself, not an under-calibration artifact.
- [ ] Quantization-aware training (QAT) as a third variant alongside FP32/PTQ INT8,
      re-run full eval suite; QAT's differentiable fake-quant path also fills the
      "INT8 white-box" gap noted in Threat models
- [ ] Expand corruption suite (JPEG compression, fog, rain, shadow, motion blur,
      perspective), categorized photometric/geometric/noise/environmental
- [ ] Second OOD dataset beyond Mapillary+DFG
- [ ] README overhaul: lead with key finding + figures, restructure around
      question -> result -> evidence -> methodology -> reproduction

## Notes
- Dataset: GTSRB (Kaggle: meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- GTSRB train images are sequential video frames of the same physical sign — a naive
  random split leaks near-duplicates between train/val. Using a track-aware split
  (group by track id, split by group) to avoid inflating accuracy.
- Frameworks: PyTorch (MPS backend available on this machine), ONNX Runtime for INT8.
