"""Train a variance-preserving diffusion (energy-parameterized DDPM) on eight-gaussians.

The variance-preserving counterpart to `train_ncsn.py`: a DDPM ε-prediction loss
where the ε-network is *derived* from a noise-conditional energy
(``ε_θ = √(1-ᾱ_t)·∇E``), then sampled with the DDPM ancestral (reverse-process)
sampler. The energy stays the primary object — the same one you could hand to a
Langevin sampler or the eval metrics.

Run:  python examples/train_diffusion.py   (CPU, ~2 minutes)
Outputs diffusion_result.png next to this script (needs the [viz] extra).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ebm


def main() -> None:
    torch.manual_seed(0)
    data = ebm.datasets.eight_gaussians(8192)

    schedule = ebm.VPSchedule(num_steps=1000, schedule="cosine")
    net = ebm.nets.NoiseConditionalMLPEnergy(dim=2, hidden=(128, 128))
    ebm.Trainer(net, ebm.VPDenoisingScoreMatching(schedule), lr=1e-3, device="cpu").fit(
        data, steps=8000, batch_size=256
    )

    samples = ebm.DDPMAncestralSampler(schedule).sample(net, torch.randn(2000, 2))
    print(f"MMD² to data = {ebm.eval.mmd(samples, data[:2000], bandwidth=0.3):.4f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    ebm.viz.plot_samples(data[:2000], ax=axes[0])
    axes[0].set_title("data")
    ebm.viz.plot_samples(samples, ax=axes[1])
    axes[1].set_title("DDPM ancestral samples")
    fig.suptitle("Variance-preserving diffusion (energy-parameterized)")
    out = Path(__file__).parent / "diffusion_result.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
