"""Deep-ensemble EBM: member disagreement as an epistemic OOD signal.

A single energy can be confidently wrong off-distribution. An ensemble of energy
networks trained on the same data agrees where it saw data and *disagrees* where
it didn't — so the variance of the member energies is an epistemic-uncertainty
score that flags OOD inputs. `EnsembleEnergy` pools the members into a mean energy
(a geometric-mean density) for sampling/scoring, and `ensemble_disagreement`
returns the per-sample variance; here it separates two-moons (in-distribution)
from a Gaussian blob (OOD) at AUROC ≈ 1.

Run:  python examples/ensemble_ood.py
Outputs ensemble_ood_result.png next to this script (needs the [viz] extra).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ebm


def _train_member(data: torch.Tensor, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    net = ebm.nets.MLPEnergy(dim=2, hidden=(128, 128))
    sampler = ebm.LangevinDynamics(step_size=0.01, steps=60)
    loss_fn = ebm.ContrastiveDivergence(
        sampler, buffer=ebm.ReplayBuffer(8192, (2,)), energy_reg=0.1
    )
    trainer = ebm.Trainer(net, loss_fn, lr=1e-3, ema_decay=0.999, device="cpu")
    trainer.fit(data, steps=3000, batch_size=256)
    return net


def main() -> None:
    torch.manual_seed(0)
    data = ebm.datasets.two_moons(8192)
    ensemble = ebm.EnsembleEnergy(*[_train_member(data, seed) for seed in range(3)])

    x_in = ebm.datasets.two_moons(2000)
    x_out = torch.randn(2000, 2) * 0.6 + torch.tensor([0.0, 3.0])  # off-manifold blob
    d_in = ebm.eval.ensemble_disagreement(ensemble, x_in)
    d_out = ebm.eval.ensemble_disagreement(ensemble, x_out)
    auroc = ebm.eval.ood_auroc(lambda z: ebm.eval.ensemble_disagreement(ensemble, z), x_in, x_out)
    print(f"mean disagreement  in-dist {d_in.mean():.3f}   OOD {d_out.mean():.3f}")
    print(f"OOD AUROC (disagreement) = {auroc:.3f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    grid = torch.linspace(-3, 4, 200)
    gy, gx = torch.meshgrid(grid, grid, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    dis = ebm.eval.ensemble_disagreement(ensemble, pts).reshape(200, 200)
    im = axes[0].imshow(dis, extent=(-3, 4, -3, 4), origin="lower", vmax=float(dis.quantile(0.98)))
    axes[0].scatter(x_in[:, 0], x_in[:, 1], s=3, color="white", alpha=0.3)
    axes[0].set_title("member disagreement (bright = uncertain / OOD)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].hist(d_in.numpy(), bins=50, alpha=0.6, label="two-moons (in)", density=True)
    axes[1].hist(d_out.numpy(), bins=50, alpha=0.6, label="Gaussian blob (OOD)", density=True)
    axes[1].set_xlabel("ensemble disagreement")
    axes[1].set_title(f"OOD AUROC = {auroc:.3f}")
    axes[1].legend()

    out = Path(__file__).parent / "ensemble_ood_result.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
