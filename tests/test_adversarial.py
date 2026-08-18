import torch
import torch.nn as nn

from src.robustness.adversarial import fgsm_attack, pgd_attack


def _toy_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 5))


def _toy_batch(n=6):
    torch.manual_seed(1)
    x = torch.rand(n, 3, 8, 8)
    y = torch.randint(0, 5, (n,))
    return x, y


def test_fgsm_respects_epsilon_bound():
    model = _toy_model()
    x, y = _toy_batch()
    epsilon = 0.03
    x_adv = fgsm_attack(model, x, y, epsilon)
    assert (x_adv - x).abs().max().item() <= epsilon + 1e-6


def test_fgsm_zero_epsilon_is_identity():
    model = _toy_model()
    x, y = _toy_batch()
    x_adv = fgsm_attack(model, x, y, epsilon=0.0)
    assert torch.allclose(x_adv, x, atol=1e-6)


def test_fgsm_output_stays_in_pixel_range():
    model = _toy_model()
    x, y = _toy_batch()
    x_adv = fgsm_attack(model, x, y, epsilon=0.5)
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0


def test_pgd_respects_epsilon_bound():
    model = _toy_model()
    x, y = _toy_batch()
    epsilon = 0.05
    x_adv = pgd_attack(model, x, y, epsilon=epsilon, alpha=0.02, steps=10)
    assert (x_adv - x).abs().max().item() <= epsilon + 1e-6


def test_pgd_output_stays_in_pixel_range():
    model = _toy_model()
    x, y = _toy_batch()
    x_adv = pgd_attack(model, x, y, epsilon=0.5, alpha=0.1, steps=5)
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0


def test_pgd_zero_epsilon_is_identity():
    model = _toy_model()
    x, y = _toy_batch()
    x_adv = pgd_attack(model, x, y, epsilon=0.0, alpha=0.1, steps=5)
    assert torch.allclose(x_adv, x, atol=1e-6)


def test_attacks_do_not_require_grad_on_output():
    model = _toy_model()
    x, y = _toy_batch()
    x_adv = pgd_attack(model, x, y, epsilon=0.05, alpha=0.02, steps=3)
    assert not x_adv.requires_grad
