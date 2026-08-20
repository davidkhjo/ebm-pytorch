"""Classifier calibration metrics (for JEM / any softmax classifier)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _bin_edges(n_bins: int, device) -> Tensor:
    return torch.linspace(0, 1, n_bins + 1, device=device)


def _bin_mask(conf: Tensor, lo: Tensor, hi: Tensor, first: bool) -> Tensor:
    # Left-open / right-closed bins, with the first bin including its left edge.
    return (conf >= lo) & (conf <= hi) if first else (conf > lo) & (conf <= hi)


@torch.no_grad()
def expected_calibration_error(probs: Tensor, labels: Tensor, n_bins: int = 15) -> float:
    """Top-label expected calibration error (Guo et al. 2017).

    Bins predictions by their confidence ``max_k p(k|x)`` and averages the gap
    between bin accuracy and bin confidence, weighted by bin population. ``0`` for
    a perfectly calibrated classifier; larger means over- or under-confident.
    """
    conf, pred = probs.max(dim=1)
    acc = (pred == labels).to(probs.dtype)
    edges = _bin_edges(n_bins, probs.device)
    ece = probs.new_zeros(())
    for i in range(n_bins):
        m = _bin_mask(conf, edges[i], edges[i + 1], first=(i == 0))
        if m.any():
            ece += m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()
    return float(ece)


@torch.no_grad()
def reliability_curve(
    probs: Tensor, labels: Tensor, n_bins: int = 15
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-bin ``(mean confidence, accuracy, count)`` for a reliability diagram."""
    conf, pred = probs.max(dim=1)
    acc = (pred == labels).to(probs.dtype)
    edges = _bin_edges(n_bins, probs.device)
    confs, accs, counts = [], [], []
    for i in range(n_bins):
        m = _bin_mask(conf, edges[i], edges[i + 1], first=(i == 0))
        confs.append(conf[m].mean() if m.any() else (edges[i] + edges[i + 1]) / 2)
        accs.append(acc[m].mean() if m.any() else probs.new_zeros(()))
        counts.append(m.sum())
    return torch.stack(confs), torch.stack(accs), torch.stack(counts)


def temperature_scale(
    logits: Tensor, labels: Tensor, *, lr: float = 0.05, max_iter: int = 200
) -> Tensor:
    """Fit a scalar temperature ``T > 0`` by NLL (Guo et al. 2017); returns ``T``.

    Divide logits by ``T`` at inference (``(logits/T).softmax(1)``) to recalibrate
    ``p(y|x)`` without changing the argmax. Fit on a **held-out** split. Optimizes
    ``log T`` with LBFGS so ``T`` stays positive without a constraint.
    """
    logits = logits.detach()  # only T is optimized, never the classifier
    log_t = torch.zeros(1, device=logits.device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

    def closure() -> Tensor:
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)  # type: ignore[arg-type]
    return log_t.exp().detach()
