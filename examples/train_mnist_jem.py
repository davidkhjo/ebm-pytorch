"""JEM at image scale: one MNIST network that classifies, detects OOD, and draws.

The same IGEBM short-run recipe as train_mnist.py, but the energy is a
classifier's ``-logsumexp(logits)`` (Grathwohl et al., 2020), so a single
network gives you:

- a digit classifier (``p(y|x)`` = softmax of the logits) — ~95% test acc,
- a density model whose energy flags out-of-distribution inputs — ~0.99
  MNIST-vs-FashionMNIST AUROC, the headline JEM capability, and
- class-conditional generation, by running Langevin on ``E(x, y)`` with
  ``energy.condition(k)``.

The classification and OOD results are strong at this budget; the conditional
samples show clear *per-class* structure (each row differs) but stay rough —
crisp digit *generation* is the expensive part and wants paper-scale budgets,
the same lesson as train_mnist.py. Two things are nonetheless essential to get
this far, both easy to get wrong (see below): ``conditional_negatives=True``
and a position-sensitive (non-pooled) classifier head.

Runs on MPS/CUDA if available (~10 min), CPU works but is slow.

Run:  python examples/train_mnist_jem.py [steps]
Outputs mnist_jem_result.png next to this script (requires the [viz] extra).
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

    x, y = ebm.datasets.mnist(return_labels=True)

    energy = ebm.ClassifierEnergy(
        ebm.nets.ConvClassifier(
            num_classes=10, in_channels=1, image_size=28, channels=(32, 64, 128)
        )
    )
    sampler = ebm.LangevinDynamics(
        step_size=10.0, noise_scale=0.005, grad_clip=0.01, steps=60, clamp=(-1, 1)
    )
    # conditional_negatives is what makes generation-on-demand work: each
    # negative chain targets a random class's E(x, y), so the per-logit
    # sampling paths used by energy.condition(k) below are actually trained.
    loss_fn = ebm.JEMLoss(
        ebm.ContrastiveDivergence(
            sampler,
            buffer=ebm.ReplayBuffer(10_000, (1, 28, 28), reinit_prob=0.05, init_fn=uniform_init),
            energy_reg=1.0,
        ),
        conditional_negatives=True,
    )

    trainer = ebm.Trainer(energy, loss_fn, lr=1e-4, ema_decay=0.999)
    print(f"training on {trainer.device} for {steps} steps")
    trainer.fit((x, y), steps=steps, batch_size=128)

    model = trainer.ema.module if trainer.ema is not None else energy
    model.eval()
    device = trainer.device

    # Classifier accuracy on the held-out split.
    x_test, y_test = ebm.datasets.mnist(train=False, return_labels=True)
    with torch.no_grad():
        preds = torch.cat(
            [model.logits(chunk.to(device)).argmax(1).cpu() for chunk in x_test.split(1000)]
        )
    acc = (preds == y_test).float().mean().item()

    # OOD detection: in-distribution digits should have lower energy than
    # FashionMNIST images the model has never seen.
    x_ood = ebm.datasets.fashion_mnist(train=False)
    auroc = ebm.eval.ood_auroc(lambda t: model(t.to(device)), x_test[:2000], x_ood[:2000])
    print(f"test accuracy: {acc:.4f}   MNIST-vs-FashionMNIST OOD AUROC: {auroc:.4f}")

    # Class-conditional generation: short-run chains on E(x, y=k), one row per
    # digit, using the same generation horizon as the unconditional example.
    rows = []
    for k in range(10):
        noise = uniform_init((8, 1, 28, 28)).to(device)
        rows.append(sampler.sample(model.condition(k), noise, steps=300).cpu())
    samples = torch.cat(rows)

    import matplotlib

    matplotlib.use("Agg")

    # 8 samples per class, stacked class-by-class, so nrow=8 puts one class per row.
    ax = ebm.viz.show_images(
        samples,
        nrow=8,
        title=(
            f"JEM class-conditional MNIST samples (row = class 0-9)\n"
            f"accuracy {acc:.3f}, OOD AUROC {auroc:.3f}"
        ),
    )
    out = Path(__file__).parent / "mnist_jem_result.png"
    ax.figure.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
