"""Recover a categorical distribution with concrete score matching — no MCMC.

`train_potts.py` *samples* a Potts lattice with Gibbs-with-Gradients. Here we do
density estimation on a tiny lattice: given samples from a known Potts
distribution, fit a flexible categorical energy by concrete score matching (the
discrete analogue of score matching — sampler-free, just single-site energy
differences) and check the recovered probabilities against the exact ground truth.
Because the 2x2, 3-colour lattice has only 81 states we can enumerate the whole
pmf and read off the fit exactly.

Run:  python examples/train_potts_concrete.py
Outputs potts_concrete_result.png next to this script (needs [viz]).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import torch
from torch import nn

import ebm

H = W = 2
K = 3
D = H * W


def all_states() -> torch.Tensor:
    return torch.stack(
        [
            torch.nn.functional.one_hot(torch.tensor(c).reshape(H, W), K).float()
            for c in itertools.product(range(K), repeat=D)
        ]
    )  # (81, 2, 2, 3)


class FlexibleCategoricalEnergy(nn.Module):
    """A generic learnable energy on one-hot (B, H, W, K) — not tied to Potts."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D * K, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)


def main() -> None:
    torch.manual_seed(0)
    states = all_states()

    # Ground truth: a Potts distribution. Enumerable, so we can sample it exactly.
    true_pmf = torch.softmax(-ebm.nets.PottsEnergy(coupling=0.9)(states), dim=0)
    data = states[torch.multinomial(true_pmf, 20000, replacement=True)]

    model = FlexibleCategoricalEnergy()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    csm = ebm.ConcreteScoreMatching()
    for step in range(2500):
        batch = data[torch.randint(0, len(data), (256,))]
        loss = csm(model, batch).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    model_pmf = torch.softmax(-model(states), dim=0)
    kl = (true_pmf * (true_pmf.log() - model_pmf.clamp_min(1e-9).log())).sum()
    print(f"KL(true || recovered) over all {K**D} states = {kl.item():.4f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    hi = float(max(true_pmf.max(), model_pmf.max())) * 1.1
    ax.plot([0, hi], [0, hi], color="#b0b0b0", ls="--", label="y = x")
    ax.scatter(true_pmf.detach(), model_pmf.detach(), s=30, color="#5c50c9", zorder=3)
    ax.set_xlabel("true probability")
    ax.set_ylabel("recovered probability (concrete SM)")
    ax.set_title(f"Categorical density recovered without MCMC\nKL = {kl.item():.4f}")
    ax.legend()
    out = Path(__file__).parent / "potts_concrete_result.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
