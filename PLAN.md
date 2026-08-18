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
- [x] Day 7 — Controlled debugging exercise: introduce 2-3 defects, reproduce failures,
      diagnose, fix, add regression tests, verify full suite
      -> 3 defects: track-split regression (caught by existing test), wrong F1 averaging
         in evaluate.summarize() (no prior coverage), swapped X/Y ROI crop coords in
         GTSRBDataset (no prior coverage). All diagnosed, fixed, 6 new regression tests
         added (each manually verified to fail on the buggy code first). Full suite:
         16 -> 22 tests, all passing. See DEBUGGING.md for the full writeup.
      -> Introduced/fixed after Day 6 results were already recorded -- no saved
         checkpoints/reports were computed with the buggy code.
- [ ] Day 8 — git init + push finished project to GitHub in one commit history

Optional / cut first if time-constrained: Mapillary generalization check, OpenCV webcam demo.

## Notes
- Dataset: GTSRB (Kaggle: meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- GTSRB train images are sequential video frames of the same physical sign — a naive
  random split leaks near-duplicates between train/val. Using a track-aware split
  (group by track id, split by group) to avoid inflating accuracy.
- Frameworks: PyTorch (MPS backend available on this machine), ONNX Runtime for INT8.
