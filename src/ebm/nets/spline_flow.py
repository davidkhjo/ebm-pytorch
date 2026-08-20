"""Rational-quadratic neural spline flow (Durkan et al. 2019)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ebm._functional import standard_normal_logprob


def _searchsorted(sorted_seq: Tensor, values: Tensor) -> Tensor:
    """Per-row bin index ``k`` with ``sorted_seq[k] <= values < sorted_seq[k+1]``."""
    return torch.searchsorted(sorted_seq, values[..., None], right=True)[..., 0] - 1


def _monotonic_rq_spline(
    inputs: Tensor,
    unnorm_widths: Tensor,
    unnorm_heights: Tensor,
    unnorm_derivs: Tensor,
    *,
    inverse: bool,
    bound: float,
    min_bin: float,
    min_deriv: float,
) -> tuple[Tensor, Tensor]:
    """Elementwise rational-quadratic transform on ``[-bound, bound]`` (identity tails).

    ``inputs`` is ``(N,)``; ``unnorm_widths``/``unnorm_heights`` are ``(N, K)`` and
    ``unnorm_derivs`` is ``(N, K-1)``. Returns ``(outputs, log|dy/dx|)``. Outside the
    interval the map is the identity (slope-1 linear tails, boundary derivatives 1),
    so this is the *unconstrained* variant (Durkan et al. 2019, eqs. 4–5).
    """
    k = unnorm_widths.shape[-1]
    inside = (inputs >= -bound) & (inputs <= bound)
    outputs = torch.where(inside, torch.zeros_like(inputs), inputs)  # identity tails
    logabsdet = torch.zeros_like(inputs)
    if not bool(inside.any()):
        return outputs, logabsdet

    inp = inputs[inside]
    uw, uh, ud = unnorm_widths[inside], unnorm_heights[inside], unnorm_derivs[inside]

    widths = min_bin + (1 - min_bin * k) * F.softmax(uw, dim=-1)
    cumwidths = F.pad(torch.cumsum(2 * bound * widths, dim=-1), (1, 0))
    cumwidths = cumwidths - bound
    cumwidths[..., 0], cumwidths[..., -1] = -bound, bound
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    heights = min_bin + (1 - min_bin * k) * F.softmax(uh, dim=-1)
    cumheights = F.pad(torch.cumsum(2 * bound * heights, dim=-1), (1, 0))
    cumheights = cumheights - bound
    cumheights[..., 0], cumheights[..., -1] = -bound, bound
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    derivs = F.pad(min_deriv + F.softplus(ud), (1, 1), value=1.0)  # slope-1 tails
    delta = heights / widths  # per-bin secant slope s_k

    knots = cumheights if inverse else cumwidths
    idx = _searchsorted(knots, inp).clamp(0, k - 1)[..., None]
    x_k = cumwidths.gather(-1, idx)[..., 0]
    w_k = widths.gather(-1, idx)[..., 0]
    y_k = cumheights.gather(-1, idx)[..., 0]
    h_k = heights.gather(-1, idx)[..., 0]
    s_k = delta.gather(-1, idx)[..., 0]
    d_k = derivs.gather(-1, idx)[..., 0]
    d_k1 = derivs.gather(-1, idx + 1)[..., 0]

    if inverse:
        dy = inp - y_k
        a = dy * (d_k1 + d_k - 2 * s_k) + h_k * (s_k - d_k)
        b = h_k * d_k - dy * (d_k1 + d_k - 2 * s_k)
        c = -s_k * dy
        disc = b.pow(2) - 4 * a * c
        theta = 2 * c / (-b - torch.sqrt(disc))  # numerically stable root
        out = theta * w_k + x_k
    else:
        theta = (inp - x_k) / w_k

    one_m = 1 - theta
    denom = s_k + (d_k1 + d_k - 2 * s_k) * theta * one_m
    if not inverse:
        out = y_k + h_k * (s_k * theta.pow(2) + d_k * theta * one_m) / denom
    deriv_num = s_k.pow(2) * (d_k1 * theta.pow(2) + 2 * s_k * theta * one_m + d_k * one_m.pow(2))
    log_deriv = torch.log(deriv_num) - 2 * torch.log(denom)

    outputs[inside] = out
    logabsdet[inside] = -log_deriv if inverse else log_deriv
    return outputs, logabsdet


class _SplineCouplingLayer(nn.Module):
    """One RQ-spline coupling: splines the ``1-mask`` half conditioned on the ``mask`` half."""

    mask: Tensor

    def __init__(self, dim: int, hidden: int, mask: Tensor, num_bins: int, bound: float):
        super().__init__()
        self.register_buffer("mask", mask)
        self.dim = dim
        self.num_bins = num_bins
        self.bound = bound
        self.min_bin = 1e-3
        self.min_deriv = 1e-3
        head = nn.Linear(hidden, dim * (3 * num_bins - 1))
        nn.init.zeros_(head.weight)  # zero last layer → near-identity spline at init
        nn.init.zeros_(head.bias)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            head,
        )

    def _params(self, conditioned: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        p = self.net(conditioned).reshape(-1, self.dim, 3 * self.num_bins - 1)
        k = self.num_bins
        return p[..., :k], p[..., k : 2 * k], p[..., 2 * k :]

    def _apply_spline(self, x: Tensor, inverse: bool) -> tuple[Tensor, Tensor]:
        uw, uh, ud = self._params(x * self.mask)
        out, lad = _monotonic_rq_spline(
            x.reshape(-1),
            uw.reshape(-1, self.num_bins),
            uh.reshape(-1, self.num_bins),
            ud.reshape(-1, self.num_bins - 1),
            inverse=inverse,
            bound=self.bound,
            min_bin=self.min_bin,
            min_deriv=self.min_deriv,
        )
        out = out.reshape_as(x)
        active = 1 - self.mask
        y = self.mask * x + active * out
        logdet = (lad.reshape_as(x) * active).sum(dim=-1)
        return y, logdet

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self._apply_spline(x, inverse=False)

    def inverse(self, y: Tensor) -> Tensor:
        return self._apply_spline(y, inverse=True)[0]


class NeuralSplineCouplingFlow(nn.Module):
    """Rational-quadratic neural spline flow, exposed as an exact-likelihood energy.

    Coupling layers whose transform is a monotonic **rational-quadratic spline**
    (Durkan et al. 2019) rather than an affine map — strictly more expressive per
    layer than `AffineCouplingFlow`, so it fits sharp, multi-modal 2D densities
    with fewer layers. Same contract as the affine flow: exact
    ``log p(x) = log N(f(x); 0, I) + log|det ∂f/∂x|`` (no partition function), so
    ``forward(x) = -log_prob(x)`` is a valid self-normalized `EnergyFn` with
    ``log Z = 0``. Operates on vector data ``(B, dim)``.

    Each spline acts on ``[-bound, bound]`` with ``num_bins`` bins and slope-1
    linear tails outside it; the conditioner MLP is zero-initialized so training
    starts from ≈identity. The spline only reshapes mass *inside* ``bound``, so
    scale the data to sit within it (or widen ``bound``). Args mirror
    `AffineCouplingFlow` plus ``num_bins`` / ``bound``.
    """

    def __init__(
        self,
        dim: int,
        n_layers: int = 6,
        hidden: int = 64,
        *,
        num_bins: int = 8,
        bound: float = 3.0,
    ):
        super().__init__()
        if dim < 2:
            raise ValueError("NeuralSplineCouplingFlow needs dim >= 2 to split coordinates")
        self.dim = dim
        layers = []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[i % 2 :: 2] = 1.0  # alternate which half is conditioned
            layers.append(_SplineCouplingLayer(dim, hidden, mask, num_bins, bound))
        self.layers = nn.ModuleList(layers)

    def transform(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Map data ``x`` to the base space, returning ``(z, log|det ∂z/∂x|)``."""
        logdet = x.new_zeros(x.shape[0])
        for layer in self.layers:
            x, ld = layer(x)
            logdet = logdet + ld
        return x, logdet

    def inverse(self, z: Tensor) -> Tensor:
        """Map base samples ``z`` back to data space."""
        for layer in reversed(self.layers):
            assert isinstance(layer, _SplineCouplingLayer)
            z = layer.inverse(z)
        return z

    def log_prob(self, x: Tensor) -> Tensor:
        z, logdet = self.transform(x)
        return standard_normal_logprob(z) + logdet

    def sample(self, n: int) -> Tensor:
        z = torch.randn(n, self.dim, device=next(self.parameters()).device)
        return self.inverse(z)

    def forward(self, x: Tensor) -> Tensor:
        return -self.log_prob(x)  # energy = -log p, so log Z = 0 exactly
