"""Goodness-of-fit tests: KSD for model selection, C2ST for two-sample fit.

Two honest demos of the goodness-of-fit metrics:

1. **Kernel Stein discrepancy** needs only a model's *score* — no samples from it,
   no partition function — so it can rank candidate energies against data. On data
   from N(0, I) we sweep the isotropic Gaussian energies ``E_a = a‖x‖²/2`` and
   confirm KSD² is minimized at the true ``a = 1``. (KSD is score-magnitude
   sensitive, which is exactly what makes it a sharp *selector* here.)

2. **Classifier two-sample test** trains a net to tell held-out data from a
   model's samples; a good fit is indistinguishable (accuracy ≈ 0.5), a wrong
   model is not. We contrast a trained EBM with a plain Gaussian on two-moons.

Run:  python examples/goodness_of_fit.py
Outputs goodness_of_fit_result.png next to this script (needs [viz]).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ebm


def main() -> None:
    torch.manual_seed(0)

    # --- 1. KSD model selection on N(0, I) data ------------------------------
    normal = torch.randn(1000, 2)
    candidates = torch.linspace(0.4, 2.2, 19)
    ksd_curve = [
        ebm.eval.kernel_stein_discrepancy(lambda z, a=a: 0.5 * a * z.pow(2).sum(dim=1), normal)
        for a in candidates
    ]
    best = candidates[int(torch.tensor(ksd_curve).argmin())].item()
    print(f"KSD-selected variance parameter a = {best:.2f}  (true a = 1.0)")

    # --- 2. C2ST: trained EBM vs a wrong model on two-moons ------------------
    train = ebm.datasets.two_moons(8192)
    test = ebm.datasets.two_moons(4000, generator=torch.Generator().manual_seed(1))
    energy = ebm.nets.MLPEnergy(dim=2, hidden=(128, 128))
    ebm.Trainer(
        energy, ebm.EnergyDiscrepancy(sigma=0.3, m_particles=16), lr=1e-3, device="cpu"
    ).fit(train, steps=6000, batch_size=256)
    sampler = ebm.LangevinDynamics(step_size=0.01, steps=500)
    good = sampler.sample(energy, torch.randn(4000, 2))
    bad = sampler.sample(lambda z: 0.5 * z.pow(2).sum(dim=1), torch.randn(4000, 2))

    c2st_good = ebm.eval.classifier_two_sample_test(test, good)
    c2st_bad = ebm.eval.classifier_two_sample_test(test, bad)
    print(f"C2ST accuracy — trained EBM {c2st_good:.2f} (≈0.5 good), Gaussian {c2st_bad:.2f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(candidates, ksd_curve, "-o", color="#5c50c9", ms=4)
    axes[0].axvline(1.0, color="#b0b0b0", ls="--", label="true a")
    axes[0].set_xlabel("candidate a")
    axes[0].set_ylabel("KSD²")
    axes[0].set_title("KSD selects the right model (no samples needed)")
    axes[0].legend()

    axes[1].scatter(test[:, 0], test[:, 1], s=4, alpha=0.25, color="#b0b0b0", label="data")
    axes[1].scatter(good[:, 0], good[:, 1], s=4, alpha=0.35, color="#5c50c9", label="EBM samples")
    axes[1].set_title(f"trained EBM on two-moons\nC2ST {c2st_good:.2f} vs Gaussian {c2st_bad:.2f}")
    axes[1].set_xlim(-2.5, 2.5)
    axes[1].set_ylim(-2, 2)
    axes[1].legend(markerscale=2)

    out = Path(__file__).parent / "goodness_of_fit_result.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
