"""Evaluation utilities: accuracy, macro-F1, per-class report, confusion matrix."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(model: nn.Module, dataset, device: torch.device, batch_size: int = 128, num_workers: int = 4):
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_preds, all_labels = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def summarize(preds: np.ndarray, labels: np.ndarray) -> dict:
    accuracy = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds)
    return {"accuracy": accuracy, "macro_f1": macro_f1, "report": report, "confusion_matrix": cm}
