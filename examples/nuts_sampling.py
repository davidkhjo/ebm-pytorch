"""NUTS: HMC that picks its own trajectory length, on Neal's funnel.

The No-U-Turn Sampler removes HMC's two hand-tuned knobs: it doubles each
trajectory until the path starts to double back (a U-turn), and it tunes the step
size to a target acceptance by dual averaging during a warmup. No `leapfrog_steps`,
no step-size sweep. Neal's funnel — a Gaussian whose width is itself a Gaussian
latent `v` — is the classic stress test: the neck is sharp where `v` is negative,
so the sampler must *lengthen* its trajectories there. We plot the samples and the
distribution of tree depths (how far NUTS doubled each draw), and print the tuned
step size, mean acceptance, and divergence count.

Run:  python examples/nuts_sampling.py
Outputs nuts_sampling_result.png next to this script (needs the [viz] extra).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ebm


def main() -> None:
    torch.manual_seed(0)
    energy = ebm.nets.FunnelEnergy(dim=2, v_scale=3.0)  # v ~ N(0, 9), x | v ~ N(0, e^v)

    sampler = ebm.NUTS(step_size=0.3, steps=300, warmup=300, target_accept=0.8)
    x = sampler.sample(energy, torch.randn(3000, 2))
    print(f"tuned step size = {sampler.step_size:.3f}")
    print(f"mean acceptance = {sampler.last_accept_rate:.3f}  (target 0.8)")
    print(
        f"v marginal std  = {x[:, 0].std():.2f}  (true 3.0; identity metric under-samples the neck)"
    )
    print(f"divergences     = {sampler.divergences}")

    # Collect tree depths over a batch of post-warmup draws (step size now frozen).
    depths = []
    xd = x
    for _ in range(40):
        xd = sampler.step(energy, xd)
        depths.append(sampler.last_tree_depth)
    depths = torch.stack(depths).reshape(-1)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(x[:, 1], x[:, 0], s=5, alpha=0.3, color="#5c50c9")
    axes[0].set_xlabel("x  (neck coordinate)")
    axes[0].set_ylabel("v  (log-scale latent)")
    axes[0].set_title("NUTS samples of Neal's funnel")
    axes[0].set_xlim(-15, 15)

    hi = int(depths.max().item())
    axes[1].hist(depths.numpy(), bins=range(hi + 2), align="left", rwidth=0.8, color="#5c50c9")
    axes[1].set_xlabel("tree depth reached")
    axes[1].set_ylabel("draws")
    axes[1].set_title("Trajectory length adapts per draw (U-turn, not a fixed length)")

    out = Path(__file__).parent / "nuts_sampling_result.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
