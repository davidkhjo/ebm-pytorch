"""Discrete lattice / RBM energies with exact enumeration."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _binary_states(n: int, *, device=None, dtype=torch.float32) -> Tensor:
    """All ``2^n`` binary vectors as a ``(2^n, n)`` tensor (for exact enumeration)."""
    idx = torch.arange(2**n, device=device)
    bits = (idx[:, None] >> torch.arange(n - 1, -1, -1, device=device)[None, :]) & 1
    return bits.to(dtype)


class IsingEnergy(nn.Module):
    """2D nearest-neighbor Ising lattice energy for binary data ``(B, H, W) -> (B,)``.

    Maps a ``{0, 1}`` lattice to spins ``s = 2x - 1`` and returns the
    ferromagnetic coupling energy ``E(x) = -J Σ_⟨i,j⟩ s_i s_j`` over right and
    down neighbors. Larger ``coupling`` ``J`` favors aligned neighbors, so
    low-energy states form ordered domains. The energy is differentiable in the
    float-relaxed input, which is exactly what `GibbsWithGradients` needs, and
    it carries no per-pixel parameters — the one scalar ``coupling`` is a
    learnable `nn.Parameter` when ``learn_coupling=True`` (recover ``J`` from
    data by contrastive divergence) and a fixed buffer otherwise.
    """

    def __init__(self, coupling: float = 0.5, learn_coupling: bool = False):
        super().__init__()
        j = torch.tensor(float(coupling))
        if learn_coupling:
            self.coupling = nn.Parameter(j)
        else:
            self.register_buffer("coupling", j)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (B, H, W) binary lattice, got {tuple(x.shape)}")
        s = 2 * x - 1
        right = (s[:, :, :-1] * s[:, :, 1:]).sum(dim=(1, 2))
        down = (s[:, :-1, :] * s[:, 1:, :]).sum(dim=(1, 2))
        return -self.coupling * (right + down)


class PottsEnergy(nn.Module):
    """2D nearest-neighbor Potts lattice energy for one-hot data ``(B, H, W, K) -> (B,)``.

    The K-color generalization of `IsingEnergy`: neighbors that share a color
    lower the energy, ``E(x) = -J Σ_⟨i,j⟩ 1[c_i = c_j]`` over right and down
    neighbors. With one-hot rows the indicator is just the dot product of
    adjacent category vectors, so the energy is differentiable in the relaxed
    input — what `CategoricalGibbsWithGradients` needs. Larger ``coupling`` ``J``
    forms same-color domains. ``coupling`` is a learnable `nn.Parameter` when
    ``learn_coupling=True`` and a fixed buffer otherwise (``Ising`` is ``K=2``).
    """

    def __init__(self, coupling: float = 0.5, learn_coupling: bool = False):
        super().__init__()
        j = torch.tensor(float(coupling))
        if learn_coupling:
            self.coupling = nn.Parameter(j)
        else:
            self.register_buffer("coupling", j)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected (B, H, W, K) one-hot lattice, got {tuple(x.shape)}")
        right = (x[:, :, :-1, :] * x[:, :, 1:, :]).sum(dim=(1, 2, 3))
        down = (x[:, :-1, :, :] * x[:, 1:, :, :]).sum(dim=(1, 2, 3))
        return -self.coupling * (right + down)


class RBM(nn.Module):
    """Bernoulli–Bernoulli restricted Boltzmann machine as a free-energy ``(B, V) -> (B,)``.

    The joint ``E(v, h) = -bᵀv - cᵀh - hᵀW v`` marginalizes over the hidden
    units in closed form, giving the free energy this module returns:

    ``F(v) = -bᵀv - Σ_j softplus(c_j + W_j·v)``,  so  ``p(v) ∝ exp(-F(v))``.

    Because ``F`` is a plain energy, it trains with the ordinary
    `ebm.ContrastiveDivergence` loss — its gradient is exactly the RBM
    maximum-likelihood gradient — using `ebm.GibbsSampler` for the block-Gibbs
    negatives. For small models the partition function is available exactly via
    `log_z`, which makes this the most closed-form-checkable EBM in the library.

    Input/output are binary visible vectors in ``{0, 1}`` of shape ``(B, V)``.
    """

    def __init__(self, n_visible: int, n_hidden: int):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_visible))
        self.c = nn.Parameter(torch.zeros(n_hidden))

    def forward(self, v: Tensor) -> Tensor:
        pre = F.linear(v, self.W, self.c)  # (B, H) = c + v Wᵀ
        return -(v @ self.b) - F.softplus(pre).sum(dim=1)

    def p_h_given_v(self, v: Tensor) -> Tensor:
        """Bernoulli means ``p(h_j = 1 | v) = σ(c_j + W_j·v)``, shape ``(B, H)``."""
        return torch.sigmoid(F.linear(v, self.W, self.c))

    def p_v_given_h(self, h: Tensor) -> Tensor:
        """Bernoulli means ``p(v_i = 1 | h) = σ(b_i + hᵀW_i)``, shape ``(B, V)``."""
        return torch.sigmoid(F.linear(h, self.W.t(), self.b))

    @torch.no_grad()
    def gibbs_step(self, v: Tensor) -> Tensor:
        """One block-Gibbs transition ``v -> h -> v'`` (both layers sampled)."""
        h = torch.bernoulli(self.p_h_given_v(v))
        return torch.bernoulli(self.p_v_given_h(h))

    @torch.no_grad()
    def log_z(self) -> Tensor:
        """Exact ``log Z`` by enumerating the smaller layer — feasible for tiny RBMs."""
        if min(self.n_visible, self.n_hidden) > 20:
            raise ValueError("exact log_z enumerates 2^min(V,H) states; keep min(V,H) <= 20")
        if self.n_hidden <= self.n_visible:
            h = _binary_states(self.n_hidden, device=self.c.device, dtype=self.c.dtype)
            # Z = Σ_h exp(cᵀh + Σ_i softplus(b_i + (Wᵀh)_i)); marginalize v exactly.
            term = h @ self.c + F.softplus(F.linear(h, self.W.t(), self.b)).sum(dim=1)
            return torch.logsumexp(term, dim=0)
        v = _binary_states(self.n_visible, device=self.b.device, dtype=self.b.dtype)
        return torch.logsumexp(-self.forward(v), dim=0)
