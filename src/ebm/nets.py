"""Ready-made energy networks.

Design choices follow standard EBM practice: smooth activations (SiLU/Swish),
no batch normalization (it breaks per-sample energies and MCMC), and optional
spectral normalization for Lipschitz control (Du & Mordatch, 2019).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm


def _binary_states(n: int, *, device=None, dtype=torch.float32) -> Tensor:
    """All ``2^n`` binary vectors as a ``(2^n, n)`` tensor (for exact enumeration)."""
    idx = torch.arange(2**n, device=device)
    bits = (idx[:, None] >> torch.arange(n - 1, -1, -1, device=device)[None, :]) & 1
    return bits.to(dtype)


def _maybe_sn(layer: nn.Module, enabled: bool) -> nn.Module:
    return spectral_norm(layer) if enabled else layer


class MLPEnergy(nn.Module):
    """MLP energy function for vector data: ``(B, dim) -> (B,)``."""

    def __init__(
        self,
        dim: int,
        hidden: Sequence[int] = (128, 128),
        spectral_norm: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = dim
        for h in hidden:
            layers += [_maybe_sn(nn.Linear(in_dim, h), spectral_norm), nn.SiLU()]
            in_dim = h
        layers.append(_maybe_sn(nn.Linear(in_dim, 1), spectral_norm))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


def _as_batch_sigma(sigma, x: Tensor) -> Tensor:
    """Normalize float / 0-dim / ``(B,)`` sigma to a ``(B,)`` tensor on ``x``'s device."""
    if not isinstance(sigma, Tensor):
        sigma = torch.tensor(float(sigma))
    sigma = sigma.to(device=x.device, dtype=x.dtype)
    if sigma.dim() == 0:
        sigma = sigma.expand(x.shape[0])
    return sigma


class _GaussianFourierFeatures(nn.Module):
    """Random Fourier embedding of ``log sigma`` (Song et al., score-SDE style)."""

    freqs: Tensor

    def __init__(self, embed_dim: int = 32, scale: float = 1.0):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even")
        self.register_buffer("freqs", torch.randn(embed_dim // 2) * scale)

    def forward(self, sigma: Tensor) -> Tensor:
        angles = 2 * torch.pi * torch.log(sigma)[:, None] * self.freqs[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=1)


class NoiseConditionalMLPEnergy(nn.Module):
    """Noise-conditional MLP energy ``(x: (B, dim), sigma: (B,)) -> (B,)``.

    The sigma embedding is concatenated to ``x`` at the input layer. Pairs with
    ``MultiSigmaDenoisingScoreMatching`` and ``AnnealedLangevinDynamics``.
    """

    def __init__(
        self,
        dim: int,
        hidden: Sequence[int] = (128, 128),
        embed_dim: int = 32,
        spectral_norm: bool = False,
    ):
        super().__init__()
        self.embed = _GaussianFourierFeatures(embed_dim)
        layers: list[nn.Module] = []
        in_dim = dim + embed_dim
        for h in hidden:
            layers += [_maybe_sn(nn.Linear(in_dim, h), spectral_norm), nn.SiLU()]
            in_dim = h
        layers.append(_maybe_sn(nn.Linear(in_dim, 1), spectral_norm))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, sigma) -> Tensor:
        sigma = _as_batch_sigma(sigma, x)
        h = torch.cat([x, self.embed(sigma)], dim=1)
        return self.net(h).squeeze(-1)


class NoiseConditionalConvEnergy(nn.Module):
    """Noise-conditional CNN energy ``(x: (B, C, H, W), sigma: (B,)) -> (B,)``.

    The sigma embedding modulates every block with FiLM (per-channel scale and
    bias) — bias-only conditioning is too weak when the noise ladder spans a
    wide variance range, as in recovery-likelihood training.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (64, 128, 256, 256),
        embed_dim: int = 32,
        spectral_norm: bool = False,
    ):
        super().__init__()
        self.embed = _GaussianFourierFeatures(embed_dim)
        self.convs = nn.ModuleList()
        self.films = nn.ModuleList()
        c_in = in_channels
        for c_out in channels:
            conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)
            self.convs.append(_maybe_sn(conv, spectral_norm))
            film = nn.Linear(embed_dim, 2 * c_out)
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)
            self.films.append(film)
            c_in = c_out
        self.act = nn.SiLU()
        self.head = _maybe_sn(nn.Linear(c_in, 1), spectral_norm)

    def forward(self, x: Tensor, sigma) -> Tensor:
        sigma = _as_batch_sigma(sigma, x)
        emb = self.embed(sigma)
        h = x
        for conv, film in zip(self.convs, self.films, strict=True):
            h = conv(h)
            scale, bias = film(emb).chunk(2, dim=-1)
            h = h * (1 + scale[:, :, None, None]) + bias[:, :, None, None]
            h = self.act(h)
        return self.head(h.mean(dim=(2, 3))).squeeze(-1)


class ConvClassifier(nn.Module):
    """Small strided CNN classifier ``(B, C, H, W) -> (B, K)`` for image JEM.

    The same trunk as `ConvEnergy` with a flattened (position-sensitive)
    K-logit head — wrap it in ``ebm.ClassifierEnergy`` to get the marginal
    energy ``E(x) = -logsumexp(logits)`` and class-conditional energies. No
    batch normalization, so per-sample energies and MCMC stay well-defined.

    The head is deliberately *not* pooled: a mean-pooled logit is a
    translation-invariant "bag of class evidence", and minimizing that
    conditional energy prefers a canvas tiled with class-typical strokes over
    a single centered digit. ``image_size`` fixes the flattened dimension.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        image_size: int = 32,
        channels: Sequence[int] = (64, 128, 256, 256),
        spectral_norm: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        size = image_size
        for c_out in channels:
            conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)
            layers += [_maybe_sn(conv, spectral_norm), nn.SiLU()]
            c_in = c_out
            size = (size + 1) // 2
        self.features = nn.Sequential(*layers)
        self.head = _maybe_sn(nn.Linear(c_in * size * size, num_classes), spectral_norm)

    def forward(self, x: Tensor) -> Tensor:
        h = self.features(x)
        return self.head(h.flatten(1))


class ConvEnergy(nn.Module):
    """Small strided CNN energy function for image data: ``(B, C, H, W) -> (B,)``.

    A reasonable baseline for 32x32 images; swap in your own architecture for
    serious image experiments.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (64, 128, 256, 256),
        spectral_norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out in channels:
            conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)
            layers += [_maybe_sn(conv, spectral_norm), nn.SiLU()]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = _maybe_sn(nn.Linear(c_in, 1), spectral_norm)

    def forward(self, x: Tensor) -> Tensor:
        h = self.features(x)
        h = h.mean(dim=(2, 3))
        return self.head(h).squeeze(-1)


class _ResBlock(nn.Module):
    """Pre-activation residual block; the second conv is zero-initialized so the
    block is the identity at init (unless spectral norm reparametrizes it)."""

    def __init__(
        self, c_in: int, c_out: int, *, spectral_norm: bool = False, downsample: bool = False
    ):
        super().__init__()
        self.act = nn.SiLU()
        self.conv1 = _maybe_sn(nn.Conv2d(c_in, c_out, 3, padding=1), spectral_norm)
        conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        if not spectral_norm:  # zero-init is incompatible with the SN parametrization
            nn.init.zeros_(conv2.weight)
            assert conv2.bias is not None
            nn.init.zeros_(conv2.bias)
        self.conv2 = _maybe_sn(conv2, spectral_norm)
        self.skip: nn.Module = (
            nn.Identity() if c_in == c_out else _maybe_sn(nn.Conv2d(c_in, c_out, 1), spectral_norm)
        )
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(self.act(x))
        h = self.conv2(self.act(h))
        out = self.skip(x) + h
        if self.downsample:
            out = F.avg_pool2d(out, 2)
        return out


class ResNetEnergy(nn.Module):
    """Residual CNN energy for images ``(B, C, H, W) -> (B,)`` (IGEBM / Improved-CD).

    The standard recipe for image EBMs: a stack of pre-activation residual blocks
    (SiLU, 3x3 convs, no normalization), average-pool downsampling between
    stages, global-pool then a linear head. The second conv of each block is
    zero-initialized so every block starts as the identity — a stable init for
    the long MCMC chains contrastive divergence needs. ``spectral_norm=True``
    wraps every conv and the head for Lipschitz control (Du & Mordatch, 2019).
    Works at any spatial size (the global pool removes the size dependence).
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (64, 128, 256),
        spectral_norm: bool = False,
    ):
        super().__init__()
        self.stem = _maybe_sn(nn.Conv2d(in_channels, channels[0], 3, padding=1), spectral_norm)
        blocks: list[nn.Module] = []
        c_in = channels[0]
        for c_out in channels:
            blocks.append(_ResBlock(c_in, c_out, spectral_norm=spectral_norm, downsample=True))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        self.act = nn.SiLU()
        self.head = _maybe_sn(nn.Linear(c_in, 1), spectral_norm)

    def forward(self, x: Tensor) -> Tensor:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        h = self.act(h).mean(dim=(2, 3))  # global average pool
        return self.head(h).squeeze(-1)


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


class FunnelEnergy(nn.Module):
    """Neal's funnel — the canonical MCMC stress test, as an energy ``(B, D) -> (B,)``.

    Coordinate 0 is the log-scale ``v ~ N(0, v_scale²)``; the remaining
    ``n = D - 1`` "neck" coordinates are ``x_i | v ~ N(0, e^v)``. The negative
    log density (up to a constant) is

    ``E(x) = ½[ v² / v_scale² + e^{-v} ‖x_neck‖² + n·v ]``.

    The geometry is deliberately vicious: at negative ``v`` the neck collapses to
    a needle no fixed step size can navigate, so single-scale samplers fail and
    the true marginals (``v ~ N(0, v_scale²)``, ``x_i | v ~ N(0, e^v)``) are the
    ground truth to check against. Differentiable, so it drives any gradient
    sampler. Reference: Neal (2003), "Slice Sampling".
    """

    def __init__(self, dim: int = 2, v_scale: float = 3.0):
        super().__init__()
        if dim < 2:
            raise ValueError("funnel needs dim >= 2 (one scale coordinate + a neck)")
        if v_scale <= 0:
            raise ValueError("v_scale must be positive")
        self.dim = dim
        self.v_scale = v_scale

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 2 or x.shape[1] != self.dim:
            raise ValueError(f"expected (B, {self.dim}), got {tuple(x.shape)}")
        v = x[:, 0]
        neck = x[:, 1:]
        n = neck.shape[1]
        neck_sq = neck.pow(2).sum(dim=1)
        return 0.5 * (v.pow(2) / self.v_scale**2 + torch.exp(-v) * neck_sq + n * v)


class GaussianMixtureEnergy(nn.Module):
    """Isotropic Gaussian-mixture energy with known modes ``(B, D) -> (B,)``.

    ``E(x) = -logsumexp_k( log w_k - ‖x - μ_k‖² / (2 σ²) )`` — the (unnormalized)
    energy of ``p(x) ∝ Σ_k w_k N(x; μ_k, σ² I)``. A closed-form multimodal target:
    with well-separated means and equal weights, a correct sampler must visit the
    modes in proportion to ``w_k``, which single-chain gradient MCMC cannot do
    across high barriers — exactly what `ParallelTempering` is for.
    """

    means: Tensor
    log_weights: Tensor

    def __init__(self, means: Tensor, weights: Sequence[float] | None = None, std: float = 1.0):
        super().__init__()
        means = torch.as_tensor(means, dtype=torch.float32)
        if means.dim() != 2:
            raise ValueError("means must be (K, D)")
        if std <= 0:
            raise ValueError("std must be positive")
        k = means.shape[0]
        if weights is None:
            weights = [1.0] * k
        if len(weights) != k or any(w <= 0 for w in weights):
            raise ValueError("need one positive weight per mode")
        self.std = std
        self.register_buffer("means", means)
        self.register_buffer("log_weights", torch.log(torch.tensor([float(w) for w in weights])))

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 2 or x.shape[1] != self.means.shape[1]:
            raise ValueError(f"expected (B, {self.means.shape[1]}), got {tuple(x.shape)}")
        sq = (x[:, None, :] - self.means[None, :, :]).pow(2).sum(dim=2)
        log_comp = self.log_weights[None, :] - sq / (2 * self.std**2)
        return -torch.logsumexp(log_comp, dim=1)


class BananaEnergy(nn.Module):
    """Curved "banana" (twisted-Gaussian) energy ``(B, 2) -> (B,)`` — Haario et al. 1999.

    A Gaussian bent along a parabola:
    ``E(x) = x₀²/(2σ₀²) + (x₁ - b(x₀² - σ₀²))²/(2σ₁²)``. The strong x₀–x₁ curvature
    defeats isotropic step sizes, so it is a standard MCMC stress test with a
    different failure mode from `FunnelEnergy` (varying scale) and
    `GaussianMixtureEnergy` (barriers). Its unique strength as a test target:
    exact i.i.d. samples are available in closed form via `exact_sample`, so a
    sampler's output can be checked directly.
    """

    def __init__(self, b: float = 0.5, sigma: tuple[float, float] = (1.0, 1.0)):
        super().__init__()
        if sigma[0] <= 0 or sigma[1] <= 0:
            raise ValueError("sigma entries must be positive")
        self.b = b
        self.sigma = sigma

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 2 or x.shape[1] != 2:
            raise ValueError(f"expected (B, 2), got {tuple(x.shape)}")
        s0, s1 = self.sigma
        warp = x[:, 1] - self.b * (x[:, 0] ** 2 - s0**2)
        return x[:, 0] ** 2 / (2 * s0**2) + warp**2 / (2 * s1**2)

    def exact_sample(self, n: int, *, generator: torch.Generator | None = None) -> Tensor:
        """Exact i.i.d. draws: sample a Gaussian, then bend it along the parabola."""
        s0, s1 = self.sigma
        u = torch.randn(n, 2, generator=generator) * torch.tensor([s0, s1])
        x = u.clone()
        x[:, 1] = u[:, 1] + self.b * (u[:, 0] ** 2 - s0**2)
        return x


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


class _CouplingLayer(nn.Module):
    """One affine coupling: transforms the ``1-mask`` half conditioned on the ``mask`` half."""

    mask: Tensor

    def __init__(self, dim: int, hidden: int, mask: Tensor):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * dim),
        )

    def _scale_shift(self, conditioned: Tensor) -> tuple[Tensor, Tensor]:
        s, t = self.net(conditioned).chunk(2, dim=-1)
        keep = 1 - self.mask
        return torch.tanh(s) * keep, t * keep  # tanh-bounded log-scale for stability

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        xm = x * self.mask
        s, t = self._scale_shift(xm)
        y = xm + (1 - self.mask) * (x * torch.exp(s) + t)
        return y, s.sum(dim=-1)  # log|det J| = sum of the active log-scales

    def inverse(self, y: Tensor) -> Tensor:
        ym = y * self.mask
        s, t = self._scale_shift(ym)
        return ym + (1 - self.mask) * ((y - t) * torch.exp(-s))


class AffineCouplingFlow(nn.Module):
    """RealNVP affine-coupling normalizing flow, exposed as an exact-likelihood energy.

    A stack of affine coupling layers with alternating masks, giving a
    tractable-density model with an exact log-likelihood and exact sampling. The
    change of variables ``log p(x) = log N(f(x); 0, I) + log|det ∂f/∂x|`` is exact
    (no partition function), so this doubles as a *self-normalized energy*:
    ``forward(x) = -log_prob(x)`` is a valid `EnergyFn` with ``log Z = 0``,
    usable anywhere in the library, and it can serve as an exact base for an EBM.
    Operates on vector data ``(B, dim)``.
    """

    def __init__(self, dim: int, n_layers: int = 6, hidden: int = 64):
        super().__init__()
        if dim < 2:
            raise ValueError("AffineCouplingFlow needs dim >= 2 to split coordinates")
        self.dim = dim
        layers = []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[i % 2 :: 2] = 1.0  # alternate which half is conditioned
            layers.append(_CouplingLayer(dim, hidden, mask))
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
            assert isinstance(layer, _CouplingLayer)
            z = layer.inverse(z)
        return z

    def log_prob(self, x: Tensor) -> Tensor:
        z, logdet = self.transform(x)
        log_base = -0.5 * z.pow(2).sum(dim=-1) - 0.5 * self.dim * math.log(2 * math.pi)
        return log_base + logdet

    def sample(self, n: int) -> Tensor:
        z = torch.randn(n, self.dim, device=next(self.parameters()).device)
        return self.inverse(z)

    def forward(self, x: Tensor) -> Tensor:
        return -self.log_prob(x)  # energy = -log p, so log Z = 0 exactly
