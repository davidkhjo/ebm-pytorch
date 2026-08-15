"""Recover an Ising coupling from lattice samples — with no MCMC in the loop.

`train_ising.py` *samples* an Ising lattice; here we do the inverse problem:
given lattices drawn at a known coupling J, recover J by maximum pseudo-likelihood.
Pseudo-likelihood is sampler-free — it only differences the energy over single
spin flips — so the whole fitting loop runs without a single Markov chain, and
the learnable `IsingEnergy` coupling converges straight to the truth.

Run:  python examples/train_ising_pseudolikelihood.py
Outputs ising_pseudolikelihood_result.png next to this script (needs [viz]).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ebm

SIZE = 12
TRUE_COUPLINGS = [0.15, 0.30, 0.45]


def main() -> None:
    torch.manual_seed(0)
    gibbs = ebm.GibbsWithGradients(steps=2000)
    pl = ebm.PseudoLikelihood()

    recovered = []
    for j in TRUE_COUPLINGS:
        # Draw training lattices at the true coupling (this is the only MCMC used,
        # and only to *make data* — the fit below uses none).
        data = gibbs.sample(
            ebm.nets.IsingEnergy(coupling=j),
            torch.bernoulli(torch.full((256, SIZE, SIZE), 0.5)),
        )
        model = ebm.nets.IsingEnergy(coupling=0.05, learn_coupling=True)
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        for _ in range(400):
            batch = data[torch.randint(0, len(data), (64,))]
            loss = pl(model, batch).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        j_hat = model.coupling.item()
        recovered.append(j_hat)
        print(f"true J = {j:.2f}   recovered J = {j_hat:.3f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    lim = (0.0, 0.55)
    ax.plot(lim, lim, color="#b0b0b0", ls="--", label="y = x")
    ax.scatter(TRUE_COUPLINGS, recovered, s=80, color="#5c50c9", zorder=3)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("true coupling J")
    ax.set_ylabel("recovered J (pseudo-likelihood)")
    ax.set_title("Ising coupling recovered without MCMC")
    ax.legend()
    out = Path(__file__).parent / "ising_pseudolikelihood_result.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
