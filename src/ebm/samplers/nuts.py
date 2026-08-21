"""No-U-Turn Sampler (NUTS): HMC that picks its own trajectory length."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ebm._functional import flat_sum as _flat_sum
from ebm.energy import EnergyFn
from ebm.samplers.base import Sampler
from ebm.utils import frozen_params

_NEG_INF = float("-inf")


def _bcast(mask: Tensor, like: Tensor) -> Tensor:
    """Reshape a ``(B,)`` mask to broadcast against ``like``'s event dims."""
    return mask.reshape(-1, *([1] * (like.dim() - 1)))


class NUTS(Sampler):
    """No-U-Turn Sampler (Hoffman & Gelman 2014), the multinomial variant.

    HMC with two things automated: the trajectory length (doubled until the path
    makes a U-turn, so it needs no ``leapfrog_steps``) and the step size (tuned to
    ``target_accept`` by dual averaging during a warmup, then frozen). Best on
    smooth, low-dimensional continuous targets; unbiased (targets exactly
    ``p ∝ exp(-E)``).

    All ``B`` chains are evolved together in **lockstep**: they build their trees to
    a shared depth each doubling, but a chain that has already met its stop
    criterion (a whole-span U-turn or a divergence) is frozen — every state write
    is masked, so a stopped chain's draw is identical to what an independent
    single-chain NUTS would have produced. Wasted leapfrogs on finished chains are
    the only cost; correctness is unaffected.

    Diagnostics after a run: ``last_accept_rate`` (mean Metropolis acceptance of
    the last draw), ``last_tree_depth`` (per-chain tree depth reached), and
    ``divergences`` (count over the sampling phase — persistently nonzero means the
    step size is too large or the geometry too sharp for an identity metric, e.g.
    Neal's funnel).

    Args:
        step_size: initial ε (a starting guess; warmup overwrites it).
        steps: default number of post-warmup draws per ``sample`` call.
        warmup: dual-averaging iterations (0 disables adaptation).
        target_accept: δ the warmup targets (0.8 is the NUTS default).
        max_depth: cap on tree depth (``2**max_depth`` leapfrogs per draw).
        max_delta_h: divergence threshold on the Hamiltonian error.
        gamma, t0, kappa: dual-averaging shrinkage / stabilization / decay constants.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        steps: int = 100,
        *,
        warmup: int = 1000,
        target_accept: float = 0.8,
        max_depth: int = 10,
        max_delta_h: float = 1000.0,
        gamma: float = 0.05,
        t0: float = 10.0,
        kappa: float = 0.75,
    ):
        super().__init__(steps)
        if not 0.0 < target_accept < 1.0:
            raise ValueError("target_accept must be in (0, 1)")
        if warmup < 0:
            raise ValueError("warmup must be >= 0")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self.step_size = step_size
        self.warmup = warmup
        self.target_accept = target_accept
        self.max_depth = max_depth
        self.max_delta_h = max_delta_h
        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa
        self.last_tree_depth: Tensor | None = None
        self.divergences = 0

    def _leapfrog(
        self, energy: EnergyFn, x: Tensor, p: Tensor, grad: Tensor, signed_eps: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """One leapfrog step with per-chain signed step ``signed_eps`` (``v·ε``)."""
        e = signed_eps.reshape(-1, *([1] * (x.dim() - 1)))
        p = p - 0.5 * e * grad
        x = x + e * p
        e_new, grad = self._energy_grad(energy, x)
        p = p - 0.5 * e * grad
        return x, p, grad, e_new

    @staticmethod
    def _no_uturn(x_minus: Tensor, x_plus: Tensor, p: Tensor) -> Tensor:
        return _flat_sum((x_plus - x_minus) * p) >= 0

    def _span_ok(self, xm: Tensor, xp: Tensor, pm: Tensor, pp: Tensor) -> Tensor:
        """Whole-span no-U-turn: the span still advances at both ends."""
        return self._no_uturn(xm, xp, pm) & self._no_uturn(xm, xp, pp)

    def _build_tree(
        self,
        energy: EnergyFn,
        x: Tensor,
        p: Tensor,
        grad: Tensor,
        v: Tensor,
        depth: int,
        eps: float,
        h0: Tensor,
        active: Tensor,
    ) -> tuple[Tensor, ...]:
        """Recursively double the trajectory; every write gated by ``active``.

        Returns ``(x⁻, p⁻, g⁻, x⁺, p⁺, g⁺, x_prop, logw, s, diverged, a_sum, n_a)``
        with log-space multinomial weight ``logw = logsumexp(H0 − H)`` over the
        subtree's leaves and validity mask ``s``.
        """
        if depth == 0:
            x1, p1, g1, e1 = self._leapfrog(energy, x, p, grad, v * eps)
            h1 = e1 + 0.5 * _flat_sum(p1.pow(2))
            d_h = h0 - h1
            finite = torch.isfinite(h1)
            over = (h1 - h0) > self.max_delta_h
            diverged = active & (~finite | over)
            valid = active & finite & ~over
            m = _bcast(active, x1)  # freeze inactive chains at their input state
            x1 = torch.where(m, x1, x)
            p1 = torch.where(m, p1, p)
            g1 = torch.where(m, g1, grad)
            logw = torch.where(valid, d_h, torch.full_like(d_h, _NEG_INF))
            a = torch.where(active, torch.exp(d_h.clamp(max=0.0)), torch.zeros_like(d_h))
            n_a = active.to(d_h.dtype)
            return (x1, p1, g1, x1, p1, g1, x1, logw, valid, diverged, a, n_a)

        xm, pm, gm, xp, pp, gp, prop1, logw1, s1, d1, a1, na1 = self._build_tree(
            energy, x, p, grad, v, depth - 1, eps, h0, active
        )
        active2 = active & s1  # only chains still valid extend a second subtree
        plus = _bcast(v > 0, xm)
        x2s = torch.where(plus, xp, xm)
        p2s = torch.where(plus, pp, pm)
        g2s = torch.where(plus, gp, gm)
        xm2, pm2, gm2, xp2, pp2, gp2, prop2, logw2, s2, d2, a2, na2 = self._build_tree(
            energy, x2s, p2s, g2s, v, depth - 1, eps, h0, active2
        )
        new_xm = torch.where(plus, xm, xm2)
        new_pm = torch.where(plus, pm, pm2)
        new_gm = torch.where(plus, gm, gm2)
        new_xp = torch.where(plus, xp2, xp)
        new_pp = torch.where(plus, pp2, pp)
        new_gp = torch.where(plus, gp2, gp)

        denom = torch.logaddexp(logw1, logw2)
        log_u = torch.log(torch.rand_like(logw1))
        replace = (log_u < (logw2 - denom)) & s2 & active2  # multinomial pick, -inf-safe
        prop = torch.where(_bcast(replace, prop1), prop2, prop1)

        no_uturn = self._span_ok(new_xm, new_xp, new_pm, new_pp)
        s = s1 & s2 & no_uturn & active
        return (
            new_xm,
            new_pm,
            new_gm,
            new_xp,
            new_pp,
            new_gp,
            prop,
            denom,
            s,
            d1 | d2,
            a1 + a2,
            na1 + na2,
        )

    def _draw(
        self, energy: EnergyFn, x0: Tensor, eps: float
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """One NUTS transition for the whole batch; returns ``(x, ᾱ, depth, diverged)``."""
        b = x0.shape[0]
        p = torch.randn_like(x0)
        e0, grad0 = self._energy_grad(energy, x0)
        h0 = e0 + 0.5 * _flat_sum(p.pow(2))

        xm = xp = x0
        pm = pp = p
        gm = gp = grad0
        x_prop = x0
        logw = torch.zeros(b, device=x0.device, dtype=x0.dtype)
        alive = torch.ones(b, device=x0.device, dtype=torch.bool)
        a_tot = torch.zeros(b, device=x0.device, dtype=x0.dtype)
        na_tot = torch.zeros(b, device=x0.device, dtype=x0.dtype)
        depth_reached = torch.zeros(b, device=x0.device, dtype=torch.long)
        diverged = torch.zeros(b, device=x0.device, dtype=torch.bool)

        for depth in range(self.max_depth):
            if not bool(alive.any()):
                break
            depth_reached += alive.long()
            v = torch.where(
                torch.rand(b, device=x0.device) < 0.5,
                torch.full((b,), -1.0, device=x0.device),
                torch.full((b,), 1.0, device=x0.device),
            )
            plus = _bcast(v > 0, xm)
            xs = torch.where(plus, xp, xm)
            ps = torch.where(plus, pp, pm)
            gs = torch.where(plus, gp, gm)
            nm, npm, ngm, npx, npp, ngp, prop_s, logw_s, s_s, d_s, a_s, na_s = self._build_tree(
                energy, xs, ps, gs, v, depth, eps, h0, alive
            )
            upd_p = _bcast(alive & (v > 0), xm)
            upd_m = _bcast(alive & (v < 0), xm)
            xp = torch.where(upd_p, npx, xp)
            pp = torch.where(upd_p, npp, pp)
            gp = torch.where(upd_p, ngp, gp)
            xm = torch.where(upd_m, nm, xm)
            pm = torch.where(upd_m, npm, pm)
            gm = torch.where(upd_m, ngm, gm)

            log_u = torch.log(torch.rand_like(logw))
            replace = (log_u < (logw_s - logw)) & s_s & alive
            x_prop = torch.where(_bcast(replace, x_prop), prop_s, x_prop)
            logw = torch.logaddexp(
                logw, torch.where(alive, logw_s, torch.full_like(logw_s, _NEG_INF))
            )
            a_tot += torch.where(alive, a_s, torch.zeros_like(a_s))
            na_tot += torch.where(alive, na_s, torch.zeros_like(na_s))
            diverged |= d_s & alive
            alive = alive & s_s & self._span_ok(xm, xp, pm, pp)

        alpha_bar = a_tot / na_tot.clamp_min(1.0)
        return x_prop.detach(), alpha_bar, depth_reached, diverged

    def step(self, energy: EnergyFn, x: Tensor) -> Tensor:
        x_next, alpha_bar, depth, diverged = self._draw(energy, x.detach(), self.step_size)
        self._last_accept = alpha_bar.mean()
        self.last_tree_depth = depth
        self.divergences += int(diverged.sum())
        return x_next

    def _dual_average(
        self, energy: EnergyFn, x: Tensor, n: int, eps0: float
    ) -> tuple[Tensor, float]:
        """Run ``n`` dual-averaging warmup draws from ``eps0``; return ``(x, ε̄)``."""
        if n == 0:
            return x, eps0
        mu = math.log(10 * eps0)
        log_eps = math.log(eps0)
        log_ebar = 0.0
        h_bar = 0.0
        for m in range(1, n + 1):
            x, alpha_bar, _, _ = self._draw(energy, x, math.exp(log_eps))
            x = x.detach()
            gap = self.target_accept - float(alpha_bar.mean())
            h_bar = (1 - 1 / (m + self.t0)) * h_bar + gap / (m + self.t0)
            log_eps = mu - math.sqrt(m) / self.gamma * h_bar
            eta = m**-self.kappa
            log_ebar = eta * log_eps + (1 - eta) * log_ebar
        return x, math.exp(log_ebar)

    def sample(
        self,
        energy: EnergyFn,
        x_init: Tensor,
        *,
        steps: int | None = None,
        return_trajectory: bool = False,
    ) -> Tensor:
        """Warm up (tuning ε), freeze, then draw. See the class docstring."""
        n_steps = self.steps if steps is None else steps
        x = x_init.detach().clone()
        self.divergences = 0
        module = energy if isinstance(energy, nn.Module) else None
        with frozen_params(module), torch.enable_grad():
            x, eps = self._dual_average(energy, x, self.warmup, self.step_size)
            self.step_size = eps  # freeze the averaged step size
            trajectory = [x.clone()] if return_trajectory else None
            for _ in range(n_steps):
                x = self.step(energy, x).detach()
                if trajectory is not None:
                    trajectory.append(x.clone())
        if trajectory is not None:
            return torch.stack(trajectory)
        return x
