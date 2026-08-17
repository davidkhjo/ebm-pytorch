"""Exact likelihoods for a score model via the probability-flow ODE.

Trains an NCSN (noise-conditional energy + multi-sigma denoising score matching)
on eight-gaussians, then computes exact per-sample log-likelihoods with the
probability-flow ODE (FFJORD change of variables — no partition function). The
likelihood cleanly separates in-distribution data from out-of-distribution
points, and its average bits/dim agrees with an independent AIS estimate.

Run:  python examples/exact_likelihood_ode.py   (CPU, ~1-2 minutes)
Outputs exact_likelihood_result.png next to this script (needs [viz]).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

import ebm

BITS = 2 * math.log(2)  # D * ln 2 for D=2


def main() -> None:
    torch.manual_seed(0)
    data = ebm.datasets.eight_gaussians(8192)
    train_sigmas = ebm.geometric_sigmas(2.0, 0.02, 20)
    net = ebm.nets.NoiseConditionalMLPEnergy(dim=2, hidden=(128, 128))
    ebm.Trainer(net, ebm.MultiSigmaDenoisingScoreMatching(train_sigmas), lr=1e-3, device="cpu").fit(
        data, steps=6000, batch_size=256
    )

    test = ebm.datasets.eight_gaussians(2000, generator=torch.Generator().manual_seed(1))
    ood = 2.5 * torch.rand(2000, 2) - 1.25  # a uniform box — off the data ring

    sigmas = ebm.geometric_sigmas(2.0, 0.01, 400)  # denser ladder for ODE accuracy
    bpd_test = -ebm.eval.pf_ode_log_likelihood(net, test, sigmas) / BITS
    bpd_ood = -ebm.eval.pf_ode_log_likelihood(net, ood, sigmas) / BITS
    print(f"PF-ODE bits/dim:  in-dist {bpd_test.mean():.3f}   OOD {bpd_ood.mean():.3f}")

    # Independent cross-check: AIS log Z at the smallest noise level.
    sigma_min = float(train_sigmas[-1])
    energy_min = lambda x: net(x, torch.full((len(x),), sigma_min))  # noqa: E731
    ais = ebm.ais_log_z(energy_min, (2,), base_scale=2.0, n_temps=300, n_chains=200)
    bpd_ais = -ebm.log_likelihood(energy_min, test, ais.log_z).mean() / BITS
    print(f"AIS bits/dim (cross-check): {bpd_ais:.3f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    lo = min(bpd_test.min().item(), bpd_ood.min().item())
    hi = max(bpd_test.quantile(0.99).item(), bpd_ood.quantile(0.99).item())
    bins = torch.linspace(lo, hi, 60).tolist()
    ax.hist(bpd_test.numpy(), bins=bins, alpha=0.6, color="#5c50c9", label="in-distribution")
    ax.hist(bpd_ood.numpy(), bins=bins, alpha=0.6, color="#c95c50", label="out-of-distribution")
    ax.axvline(bpd_ais.item(), color="#333", ls="--", label=f"AIS mean ({bpd_ais:.2f})")
    ax.set_xlabel("bits / dim (lower = more likely)")
    ax.set_ylabel("count")
    ax.set_title("Exact PF-ODE likelihoods separate in-dist from OOD")
    ax.legend()
    out = Path(__file__).parent / "exact_likelihood_result.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
