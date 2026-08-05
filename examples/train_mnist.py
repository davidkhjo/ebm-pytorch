"""An image-scale EBM: the IGEBM short-run recipe on MNIST.

Persistent contrastive divergence with the practitioner constants from Du &
Mordatch (2019), exactly as documented in the README: huge Langevin step,
cold decoupled noise, gradient clipping, replay buffer with 5% reinit, and
energy regularization. Generation reuses the training dynamics (short-run
chains from uniform noise) — with short-run EBMs, training *is* the sampler,
which is why this recipe demos well at small budgets. (Diffusion recovery
likelihood is the stabler *training* objective at scale, but its negatives
start near data, so learning to generate from pure noise needs paper-scale
budgets — see docs/training.md.)

Runs on MPS/CUDA if available (~5 min), CPU works but is slow.

Run:  python examples/train_mnist.py [steps]
Outputs mnist_result.png next to this script (requires the [viz] extra).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import ebm


def uniform_init(shape):
    return torch.rand(shape) * 2 - 1


def main() -> None:
    torch.manual_seed(0)
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

    x = ebm.datasets.mnist()  # (60000, 1, 28, 28) in [-1, 1]

    energy = ebm.nets.ConvEnergy(in_channels=1, channels=(32, 64, 128))
    sampler = ebm.LangevinDynamics(
        step_size=10.0, noise_scale=0.005, grad_clip=0.01, steps=60, clamp=(-1, 1)
    )
    loss_fn = ebm.ContrastiveDivergence(
        sampler,
        buffer=ebm.ReplayBuffer(10_000, (1, 28, 28), reinit_prob=0.05, init_fn=uniform_init),
        energy_reg=1.0,
    )

    trainer = ebm.Trainer(energy, loss_fn, lr=1e-4, ema_decay=0.999)
    print(f"training on {trainer.device} for {steps} steps")
    trainer.fit(x, steps=steps, batch_size=128)

    model = trainer.ema.module if trainer.ema is not None else energy
    model.eval()
    # Short-run generation: the training dynamics from fresh noise, run a few
    # times longer than the training horizon to refine the EMA model's samples.
    # Don't over-train instead: past ~10k steps the energy sharpens beyond what
    # 60-step chains can track and the energy_gap (and samples) degrade.
    noise = uniform_init((64, 1, 28, 28)).to(trainer.device)
    samples = sampler.sample(model, noise, steps=300).cpu()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for ax, img in zip(axes.flat, samples, strict=True):
        ax.imshow(img[0].clamp(-1, 1) * 0.5 + 0.5, cmap="gray_r", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle("MNIST digits from a short-run EBM (IGEBM recipe)", fontsize=12)
    fig.tight_layout()
    out = Path(__file__).parent / "mnist_result.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
