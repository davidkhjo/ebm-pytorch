"""Hard-target sampling: parallel tempering vs a single chain, with diagnostics.

A single Langevin chain started in one mode of a well-separated mixture stays
trapped there; parallel tempering crosses the barrier and recovers both modes.
We then run several independent chains and report the ESS / split-R̂ convergence
diagnostics — the honest way to tell whether a sampler actually mixed.

Run:  python examples/sampling_hard_targets.py
Outputs sampling_hard_targets_result.png next to this script (needs [viz]).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ebm


def main() -> None:
    torch.manual_seed(0)

    means = torch.tensor([[-4.0], [4.0]])
    energy = ebm.nets.GaussianMixtureEnergy(means, std=1.0)
    x0 = -4.0 + 0.3 * torch.randn(2000, 1)  # every chain starts in the LEFT mode

    single = ebm.LangevinDynamics(step_size=0.08, steps=1000).sample(energy, x0.clone())
    base = ebm.LangevinDynamics(step_size=0.08, steps=1)
    pt = ebm.ParallelTempering(base, temperatures=(1.0, 4.0, 16.0, 64.0), steps=1000)
    tempered = pt.sample(energy, x0.clone())

    print(f"single chain   P(x>0) = {(single > 0).float().mean():.3f}  (trapped)")
    print(f"tempered chain P(x>0) = {(tempered > 0).float().mean():.3f}  (both modes)")
    print("swap rates: " + ", ".join(f"{r:.2f}" for r in pt.swap_rates()))

    # Convergence diagnostics: many chains on a standard normal, drawn as a
    # (n_chains, n_samples, dim) trajectory tensor.
    traj = ebm.MALA(step_size=0.5, steps=400).sample(
        lambda x: 0.5 * x.pow(2).sum(1), torch.randn(16, 1), return_trajectory=True
    )
    chains = traj.transpose(0, 1)  # (16, 401, 1)
    total = chains.shape[0] * chains.shape[1]
    print(f"ESS   = {ebm.eval.effective_sample_size(chains).item():.0f} of {total}")
    print(f"R-hat = {ebm.eval.split_rhat(chains).item():.3f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, s, title in ((axes[0], single, "single chain"), (axes[1], tempered, "tempered")):
        ax.hist(s.flatten().numpy(), bins=60, range=(-7, 7), color="#5c50c9", alpha=0.8)
        ax.set_title(f"{title}  —  P(x>0)={(s > 0).float().mean():.2f}")
        ax.set_xlabel("x")
    fig.suptitle("Escaping a trapped mode: single Langevin chain vs parallel tempering")
    out = Path(__file__).parent / "sampling_hard_targets_result.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
