"""Fit two-moons with a rational-quadratic neural spline flow (Durkan et al. 2019).

`NeuralSplineCouplingFlow` is a drop-in for `AffineCouplingFlow` whose per-layer
transform is a monotonic rational-quadratic spline instead of an affine map —
strictly more expressive, so it carves the sharp two-moons crescents with fewer
layers and a lower held-out NLL. Same self-normalized contract: exact `log_prob`
(no MCMC), one-pass sampling, and `forward(x) = -log_prob(x)` is a valid energy.
We train an affine flow and a spline flow with the *same* layer budget and print
both held-out NLLs so the gap is visible.

Run:  python examples/train_spline_flow.py
Outputs spline_flow_result.png next to this script (requires the [viz] extra).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ebm


def _train(flow: torch.nn.Module, data: torch.Tensor, steps: int = 4000) -> torch.nn.Module:
    opt = torch.optim.Adam(flow.parameters(), lr=3e-3)
    for _ in range(steps):
        batch = data[torch.randint(0, len(data), (256,))]
        loss = -flow.log_prob(batch).mean()  # exact negative log-likelihood
        opt.zero_grad()
        loss.backward()
        opt.step()
    return flow


def main() -> None:
    torch.manual_seed(0)
    data = ebm.datasets.two_moons(8192)
    test = ebm.datasets.two_moons(4000, generator=torch.Generator().manual_seed(1))

    affine = _train(ebm.nets.AffineCouplingFlow(dim=2, n_layers=6, hidden=64), data)
    spline = _train(
        ebm.nets.NeuralSplineCouplingFlow(dim=2, n_layers=6, num_bins=8, bound=3.0), data
    )

    for name, flow in (("affine", affine), ("spline", spline)):
        print(f"{name:7s} held-out NLL = {-flow.log_prob(test).mean().item():.3f} nats")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lin = torch.linspace(-2.5, 2.5, 200)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ebm.viz.plot_samples(data[:2000], ax=axes[0])
    axes[0].set_title("data")
    for ax, (name, flow) in zip(axes[1:], (("affine", affine), ("spline", spline)), strict=True):
        with torch.no_grad():
            dens = flow.log_prob(grid).exp().reshape(200, 200)
        ax.imshow(
            dens, extent=(-2.5, 2.5, -2.5, 2.5), origin="lower", vmax=float(dens.quantile(0.98))
        )
        ax.set_title(f"{name} flow density (exact)")
    fig.suptitle("Affine vs rational-quadratic spline coupling — same 6-layer budget")
    out = Path(__file__).parent / "spline_flow_result.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
