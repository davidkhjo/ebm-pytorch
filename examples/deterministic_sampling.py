"""Three ways to sample a trained score model: annealed Langevin, PF-ODE, PC.

Trains an NCSN (noise-conditional energy + multi-sigma denoising score matching)
on eight-gaussians, then generates with the stochastic annealed-Langevin sampler,
the deterministic probability-flow ODE, and the predictor-corrector. The ODE is
reproducible — the same initial noise always gives the same samples — which the
Langevin samplers cannot offer.

Run:  python examples/deterministic_sampling.py   (CPU, ~2 minutes)
Outputs deterministic_sampling_result.png next to this script (needs [viz]).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ebm


def main() -> None:
    torch.manual_seed(0)
    data = ebm.datasets.eight_gaussians(8192)

    train_sigmas = ebm.geometric_sigmas(2.0, 0.02, 20)
    net = ebm.nets.NoiseConditionalMLPEnergy(dim=2, hidden=(128, 128))
    ebm.Trainer(net, ebm.MultiSigmaDenoisingScoreMatching(train_sigmas), lr=1e-3, device="cpu").fit(
        data, steps=8000, batch_size=256
    )

    # Denser ladder for sampling (the conditional net generalizes across sigma).
    sigmas = ebm.geometric_sigmas(2.0, 0.02, 200)
    sigma_max = float(sigmas[0])

    torch.manual_seed(1)
    langevin = ebm.AnnealedLangevinDynamics(sigmas, step_size=3e-3, steps_per_sigma=20).sample(
        net, torch.randn(2000, 2) * sigma_max
    )
    ode = ebm.ProbabilityFlowODE(sigmas)
    ode_samples = ode.sample(net, torch.randn(2000, 2) * sigma_max)
    pc = ebm.PredictorCorrector(sigmas, n_corrector=1).sample(net, torch.randn(2000, 2) * sigma_max)

    x0 = torch.randn(2000, 2) * sigma_max
    deterministic = torch.allclose(ode.sample(net, x0), ode.sample(net, x0))
    print(f"probability-flow ODE deterministic: {deterministic}")
    for name, s in (
        ("annealed Langevin", langevin),
        ("PF-ODE", ode_samples),
        ("predictor-corrector", pc),
    ):
        print(f"{name:20} MMD² to data = {ebm.eval.mmd(s, data[:2000], bandwidth=0.3):.4f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("annealed Langevin", langevin),
        ("PF-ODE (deterministic)", ode_samples),
        ("predictor-corrector", pc),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharex=True, sharey=True)
    for ax, (name, s) in zip(axes, panels, strict=True):
        ax.scatter(data[:2000, 0], data[:2000, 1], s=4, alpha=0.15, color="#b0b0b0")
        ax.scatter(s[:, 0], s[:, 1], s=4, alpha=0.4, color="#5c50c9")
        ax.set_title(name, fontsize=11)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Sampling a trained NCSN — gray: data, purple: samples")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(__file__).parent / "deterministic_sampling_result.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
