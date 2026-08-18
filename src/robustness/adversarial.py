"""FGSM and PGD adversarial attacks, operating on raw [0,1] pixel-space tensors.

Hand-rolled rather than via Foolbox/ART: both attacks are a few lines of autograd and this
avoids pulling in a heavier dependency with its own version-compatibility surface, while
staying true to the standard formulations (Goodfellow et al. 2015 for FGSM, Madry et al.
2018 for PGD).
"""

import torch
import torch.nn.functional as F


def fgsm_attack(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, epsilon: float) -> torch.Tensor:
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    (grad,) = torch.autograd.grad(loss, x)
    x_adv = x + epsilon * grad.sign()
    return x_adv.clamp(0.0, 1.0).detach()


def pgd_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    x_orig = x.clone().detach()
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        (grad,) = torch.autograd.grad(loss, x_adv)
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + epsilon), x_orig - epsilon)
        x_adv = x_adv.clamp(0.0, 1.0).detach()
    return x_adv
