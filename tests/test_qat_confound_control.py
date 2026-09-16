import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.eval_qat_confound_control import EPSILONS, PGD_STEPS, whitebox_pgd
from src.robustness.adversarial import pgd_attack


def _toy_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 5))


def _toy_loader(n=8):
    torch.manual_seed(1)
    x = torch.rand(n, 3, 8, 8)
    y = torch.randint(0, 5, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_whitebox_pgd_returns_one_entry_per_epsilon():
    model = _toy_model()
    loader = _toy_loader()
    results = whitebox_pgd(model, loader)

    assert [r["epsilon"] for r in results] == EPSILONS
    assert all(0.0 <= r["accuracy"] <= 1.0 for r in results)


def test_whitebox_pgd_aggregates_correctly_across_batches():
    """Recompute one epsilon's accuracy independently (same attack calls, manual
    accumulation) and check it matches whitebox_pgd's own total/correct bookkeeping."""
    model = _toy_model()
    loader = _toy_loader()
    epsilon = EPSILONS[-1]

    total, correct = 0, 0
    for images, labels in loader:
        adv = pgd_attack(model, images, labels, epsilon, alpha=epsilon / 4, steps=PGD_STEPS)
        with torch.no_grad():
            preds = model(adv).argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    expected_accuracy = correct / total

    result = whitebox_pgd(model, loader)[-1]
    assert result["epsilon"] == epsilon
    assert result["accuracy"] == expected_accuracy
