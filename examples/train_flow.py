"""Fit two-moons with a RealNVP normalizing flow — exact likelihood, exact samples.

Unlike the EBMs elsewhere in this repo, a normalizing flow is *self-normalized*:
its `log_prob` is exact (no partition function, no MCMC) and it samples in one
forward pass. Because `AffineCouplingFlow.forward(x) = -log_prob(x)` is a valid
energy with `log Z = 0`, the same `viz.energy_contour` used for EBMs renders its
density directly.

Run:  python examples/train_flow.py
Outputs flow_result.png next to this script (requires the [viz] extra).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ebm


def main() -> None:
    torch.manual_seed(0)
    data = ebm.datasets.two_moons(8192)
    test = ebm.datasets.two_moons(4000, generator=torch.Generator().manual_seed(1))

    flow = ebm.nets.AffineCouplingFlow(dim=2, n_layers=10, hidden=64)
    opt = torch.optim.Adam(flow.parameters(), lr=3e-3)
    for step in range(5000):
        batch = data[torch.randint(0, len(data), (256,))]
        loss = -flow.log_prob(batch).mean()  # exact negative log-likelihood
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            print(f"step {step:4d}  NLL {loss.item():.3f}")

    print(f"held-out NLL = {-flow.log_prob(test).mean().item():.3f} nats")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ebm.viz.plot_samples(data[:2000], ax=axes[0])
    axes[0].set_title("data")

    # Exact density on a grid (clip the color range so far-field flow extrapolation
    # spikes don't wash out the data region).
    lin = torch.linspace(-2.5, 2.5, 200)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    with torch.no_grad():
        dens = flow.log_prob(grid).exp().reshape(200, 200)
    axes[1].imshow(
        dens, extent=(-2.5, 2.5, -2.5, 2.5), origin="lower", vmax=float(dens.quantile(0.98))
    )
    axes[1].set_title("learned density (exact)")

    ebm.viz.plot_samples(flow.sample(2000), ax=axes[2])
    axes[2].set_title("flow samples")

    out = Path(__file__).parent / "flow_result.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
