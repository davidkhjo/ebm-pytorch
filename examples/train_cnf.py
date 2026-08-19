"""Fit two-moons with a continuous normalizing flow (FFJORD) — exact likelihood by ODE.

The trainable generalization of `eval.pf_ode_log_likelihood`: a neural velocity
field defines an ODE from data to a Gaussian base, and the exact log-density comes
from the instantaneous change of variables (Hutchinson trace), all in pure torch
(fixed-step RK4, direct backprop). Trains by maximum likelihood, samples in one
reverse pass.

Run:  python examples/train_cnf.py   (CPU, ~2 minutes)
Outputs cnf_result.png next to this script (needs the [viz] extra).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ebm


def main() -> None:
    torch.manual_seed(0)
    data = ebm.datasets.two_moons(8192)
    test = ebm.datasets.two_moons(4000, generator=torch.Generator().manual_seed(1))

    cnf = ebm.nets.ContinuousNormalizingFlow(dim=2, hidden=(64, 64), n_steps=20)
    opt = torch.optim.Adam(cnf.parameters(), lr=3e-3)
    cnf.train()
    for step in range(1500):
        batch = data[torch.randint(0, len(data), (256,))]
        loss = -cnf.log_prob(batch).mean()  # exact negative log-likelihood
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 300 == 0:
            print(f"step {step:4d}  NLL {loss.item():.3f}")

    cnf.eval()
    print(f"held-out NLL = {-cnf.log_prob(test).mean().item():.3f} nats")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ebm.viz.plot_samples(data[:2000], ax=axes[0])
    axes[0].set_title("data")

    lin = torch.linspace(-2.5, 2.5, 80)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    dens = cnf.log_prob(grid).detach().exp().reshape(80, 80)
    axes[1].imshow(
        dens, extent=(-2.5, 2.5, -2.5, 2.5), origin="lower", vmax=float(dens.quantile(0.98))
    )
    axes[1].set_title("learned density (exact, by ODE)")

    ebm.viz.plot_samples(cnf.sample(2000).detach(), ax=axes[2])
    axes[2].set_title("CNF samples")

    out = Path(__file__).parent / "cnf_result.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
