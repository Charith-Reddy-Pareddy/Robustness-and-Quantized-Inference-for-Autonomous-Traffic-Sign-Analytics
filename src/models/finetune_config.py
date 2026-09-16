"""Shared fine-tuning protocol for QAT and its FP32 equal-training control.

A single source of truth for epochs/lr/batch-size/seed so the two scripts that use them
(scripts/train_qat.py, scripts/train_fp32_extra_training.py) can't silently drift apart --
the entire point of the FP32 control is that every hyperparameter matches QAT exactly
except the presence of fake quantization.
"""

FINE_TUNE_EPOCHS = 3
FINE_TUNE_LR = 1e-4
FINE_TUNE_BATCH_SIZE = 128
FINE_TUNE_SEED = 42
