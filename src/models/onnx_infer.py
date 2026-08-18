"""Inference helper for ONNX Runtime sessions, mirroring src/models/evaluate.predict."""

import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader


def predict_onnx(onnx_path, dataset, batch_size: int = 128, num_workers: int = 0):
    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_preds, all_labels = [], []
    for images, labels in loader:
        logits = sess.run(None, {input_name: images.numpy().astype(np.float32)})[0]
        all_preds.append(logits.argmax(axis=1))
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)
