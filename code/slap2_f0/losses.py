"""Heteroscedastic likelihoods for the residual target (movie - activity).

All take the target y, predicted mean mu and variance var (same shape) and a
weight/mask w (0 = ignore), and return the weighted mean negative
log-likelihood per element (constants dropped).
"""

from __future__ import annotations

import math

import torch


def beta_nll(y: torch.Tensor, mu: torch.Tensor, var: torch.Tensor, w: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    """Gaussian NLL with each element weighted by var**beta (detached), so
    the variance head does not stall learning of the mean (Seitzer et al.
    2022). beta = 0 is the plain Gaussian NLL."""
    var = var.clamp_min(1e-8)
    nll = 0.5 * (torch.log(var) + (y - mu) ** 2 / var)
    if beta:
        nll = nll * var.detach() ** beta
    return _wmean(nll, w)


def student_t_nll(y: torch.Tensor, mu: torch.Tensor, var: torch.Tensor, w: torch.Tensor, nu: torch.Tensor | float = 5.0) -> torch.Tensor:
    """Student-t NLL with scale**2 = var * (nu - 2) / nu, so var is the
    variance for nu > 2. Robust to the positive tail of unfitted events."""
    nu = torch.as_tensor(nu, dtype=y.dtype, device=y.device)
    scale2 = (var * (nu - 2) / nu).clamp_min(1e-8)
    z2 = (y - mu) ** 2 / scale2
    nll = (
        0.5 * torch.log(scale2)
        + 0.5 * (nu + 1) * torch.log1p(z2 / nu)
        - torch.lgamma((nu + 1) / 2)
        + torch.lgamma(nu / 2)
        + 0.5 * torch.log(nu * math.pi)
    )
    return _wmean(nll, w)


def _wmean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    w = w.to(x.dtype)
    return (x * w).sum() / w.sum().clamp_min(1.0)
