# Robustness and Quantized Inference for Autonomous Traffic Sign Analytics

Does INT8 quantization for real-time deployment compromise a traffic-sign classifier's
robustness to visual corruptions and adversarial attacks — even when clean accuracy is
preserved?

Full write-up, methodology, and results: [reports/ROBUSTNESS_REPORT.md](reports/ROBUSTNESS_REPORT.md)

## TL;DR

Two architectures (a from-scratch CNN and a fine-tuned MobileNetV2) were trained on
GTSRB, benchmarked against corruptions (blur/noise/rotation/brightness) and adversarial
attacks (FGSM/PGD), then quantized to INT8 via ONNX Runtime and re-tested. The hypothesis
that quantization would hurt both corruption and adversarial robustness was only
**partially confirmed**: adversarial vulnerability was *not* increased by quantization
for either architecture, while corruption robustness held for the CNN but degraded for
MobileNetV2 specifically at high severity. INT8 delivered its expected deployment
benefits regardless: ~2x faster inference, ~3.4x smaller model files.

A follow-up generalization check on a different dataset (Mapillary+DFG) found something
arguably bigger: both models generalize far worse than their GTSRB numbers suggest
(baseline CNN drops 44.8 percentage points, MobileNetV2 drops 28.4pp) — but the FP32-vs-INT8
gap stays just as small under that distribution shift as it was on GTSRB.

## Project structure

```
src/
  data/          dataset loading, track-aware train/val split, transforms
  models/        architectures, training loop, evaluation, ONNX/normalization wrapper
  robustness/    image corruptions, FGSM/PGD adversarial attacks
  quantization/  ONNX export + static INT8 quantization
  viz/           Grad-CAM
scripts/         one script per pipeline stage (training, evaluation, plotting)
tests/           pytest suite
reports/         all generated figures, metrics, and the full written report
```

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires a Kaggle API token (`~/.kaggle/kaggle.json`) to download
[GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign):

```bash
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign -p data/raw --unzip
```

Then, in order:

```bash
python scripts/eda.py                       # exploratory data analysis
python scripts/run_experiments.py            # train both architectures, 3 seeds each
python scripts/eval_corruptions.py           # FP32 corruption robustness
python scripts/gradcam_demo.py               # Grad-CAM visualizations
python scripts/eval_adversarial.py           # FP32 adversarial robustness (FGSM/PGD)
python scripts/quantize_models.py            # export to ONNX + static INT8 quantization
python scripts/eval_corruptions_int8.py      # INT8 corruption robustness
python scripts/eval_transfer_attacks.py      # INT8 transfer-attack evaluation
python scripts/benchmark_latency.py          # FP32 vs INT8 latency/size
python scripts/eval_mapillary_generalization.py  # generalization check (see below)
```

The generalization check needs the [Mapillary+DFG dataset](https://www.kaggle.com/datasets/nomihsa965/traffic-signs-dataset-mapillary-and-dfg)
(~12GB unzipped) downloaded to `data/mapillary/`:

```bash
kaggle datasets download -d nomihsa965/traffic-signs-dataset-mapillary-and-dfg -p data/mapillary --unzip
```

Plotting scripts (`plot_*.py`) regenerate the figures in `reports/` from the saved JSON
results. Run the test suite with `pytest`.


## Data sources

- Primary: [GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
  (German Traffic Sign Recognition Benchmark)
- Generalization check: [Traffic Signs Dataset (Mapillary and DFG)](https://www.kaggle.com/datasets/nomihsa965/traffic-signs-dataset-mapillary-and-dfg),
  a Kaggle mirror combining crops from the Mapillary Traffic Sign Dataset and the DFG
  Traffic Sign Data Set, refined for the Africa region (76 classes). GTSRB's 43 classes
  were mapped to 23 of these by semantic meaning — see `src/data/mapillary_mapping.py`.
- The OpenCV webcam demo was scoped as lowest-priority/no-research-value in the original
  project spec and was not attempted.
