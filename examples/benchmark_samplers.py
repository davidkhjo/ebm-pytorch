"""Benchmark MCMC samplers on the banana, scored against exact draws.

The banana target ships an exact sampler, so we can rank samplers by how well
their output actually matches the truth (MMD²) alongside the usual MCMC
diagnostics — effective sample size, split-R̂, acceptance rate, and wall time.
This is the reproducible, ground-truthed version of the diagnostics shown in
`sampling_hard_targets.py`.

Run:  python examples/benchmark_samplers.py   (CPU, ~1-2 minutes)
Outputs benchmark_samplers_result.png next to this script (needs [viz]).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

import ebm

N_CHAINS = 2000
STEPS = 400


def evaluate(name, sampler, energy, exact):
    torch.manual_seed(0)
    x0 = torch.randn(N_CHAINS, 2)
    t0 = time.perf_counter()
    traj = sampler.sample(energy, x0, steps=STEPS, return_trajectory=True)
    seconds = time.perf_counter() - t0

    final = traj[-1]  # (N_CHAINS, 2)
    diag = traj[:, :16, :].transpose(0, 1)  # 16 chains × (STEPS+1) draws for diagnostics
    return {
        "name": name,
        "ess": ebm.eval.effective_sample_size(diag).mean().item(),
        "rhat": ebm.eval.split_rhat(diag).max().item(),
        "accept": sampler.last_accept_rate,
        "mmd2": ebm.eval.mmd(final[:3000], exact[:3000]),
        "seconds": seconds,
        "final": final,
    }


def main() -> None:
    # A sharply curved, thin banana: hard enough that the samplers actually separate.
    energy = ebm.nets.BananaEnergy(b=1.0, sigma=(1.0, 0.5))
    exact = energy.exact_sample(3000, generator=torch.Generator().manual_seed(1))

    samplers = {
        "ULA": ebm.LangevinDynamics(step_size=0.02),
        "MALA": ebm.MALA(step_size=0.05),
        "HMC": ebm.HMC(step_size=0.1, leapfrog_steps=15),
        "Underdamped": ebm.UnderdampedLangevin(step_size=0.03, friction=1.0),
        "Preconditioned": ebm.PreconditionedLangevin(
            precond=torch.tensor([1.0, 0.25]), step_size=0.02
        ),
    }
    results = [evaluate(name, s, energy, exact) for name, s in samplers.items()]

    print("| Sampler | ESS | R-hat | accept | MMD² ↓ | time (s) |")
    print("|---|---|---|---|---|---|")
    for r in results:
        acc = "—" if r["accept"] is None else f"{r['accept']:.2f}"
        print(
            f"| {r['name']} | {r['ess']:.0f} | {r['rhat']:.3f} | {acc} | "
            f"{r['mmd2']:.5f} | {r['seconds']:.1f} |"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(results), figsize=(3 * len(results), 3.2), sharex=True, sharey=True
    )
    for ax, r in zip(axes, results, strict=True):
        ax.scatter(exact[:, 0], exact[:, 1], s=4, alpha=0.2, color="#b0b0b0")
        ax.scatter(r["final"][:2000, 0], r["final"][:2000, 1], s=4, alpha=0.3, color="#5c50c9")
        ax.set_title(f"{r['name']}\nMMD² {r['mmd2']:.4f}", fontsize=10)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1.5, 4)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Samplers on the banana — gray: exact draws, purple: sampler output")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(__file__).parent / "benchmark_samplers_result.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
