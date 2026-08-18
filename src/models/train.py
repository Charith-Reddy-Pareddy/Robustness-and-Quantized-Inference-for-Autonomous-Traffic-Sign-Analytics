"""Generic training loop shared by the baseline CNN and the transfer-learning model."""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    epochs: int = 12
    lr: float = 1e-3
    batch_size: int = 128
    seed: int = 42
    patience: int = 3


def _run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(mode=training)
    criterion = nn.CrossEntropyLoss()
    total_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        all_preds.append(logits.argmax(dim=1).detach().cpu())
        all_labels.append(labels.detach().cpu())
    preds = torch.cat(all_preds).numpy()
    labels_arr = torch.cat(all_labels).numpy()
    macro_f1 = f1_score(labels_arr, preds, average="macro")
    return total_loss / len(loader.dataset), macro_f1


def train_model(
    model: nn.Module,
    train_ds,
    val_ds,
    device: torch.device,
    config: TrainConfig,
    ckpt_path: Path,
    num_workers: int = 4,
) -> list[dict]:
    set_seed(config.seed)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, num_workers=num_workers
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    history = []
    best_val_f1 = -1.0
    epochs_without_improvement = 0
    for epoch in range(config.epochs):
        train_loss, train_f1 = _run_epoch(model, train_loader, device, optimizer)
        val_loss, val_f1 = _run_epoch(model, val_loader, device, optimizer=None)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_macro_f1": train_f1,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
            }
        )
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping at epoch {epoch} (best val_f1={best_val_f1:.4f})")
                break
    return history
