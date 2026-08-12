"""Energy-based OOD at color scale: CIFAR-10 in-distribution, CIFAR-100 out.

A classifier is secretly an EBM — its marginal energy is `-logsumexp(logits)`
(Grathwohl et al., 2020) — so one `ConvClassifier` gives both a label and an
out-of-distribution score. We train on CIFAR-10 and flag CIFAR-100 images as
OOD purely by their energy, with no generation involved.

Training is cross-entropy-dominant (a light contrastive-divergence term shapes
the energy into a usable density) — the stable regime, in contrast to the
fragile generative CD used for the MNIST-JEM *generation* demo. See
`train_mnist_jem.py` for the generative story and its budget lesson.

Honest expectations (8000 steps): the classifier reaches ~61% CIFAR-10 test
accuracy — a real classifier — but the OOD AUROC is only ~0.57. CIFAR-10 vs
CIFAR-100 is a *near-OOD* pair (both are natural 32x32 objects), which is
genuinely hard: the energy carries a weak-but-real signal, far below the ~0.99
of MNIST-vs-FashionMNIST. A *far-OOD* pair like CIFAR-10 vs SVHN separates much
more cleanly, but SVHN ships as MATLAB `.mat` files and this library stays
torchvision-free. The demo is about the method and API at color scale, not a
state-of-the-art OOD number.

Runs on MPS/CUDA if available (~20-40 min), CPU is impractical.

Run:  python examples/train_cifar_ood.py [steps]
Outputs cifar_ood_result.png next to this script (requires the [viz] extra).
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

    x, y = ebm.datasets.cifar10(return_labels=True)

    energy = ebm.ClassifierEnergy(
        ebm.nets.ConvClassifier(num_classes=10, in_channels=3, image_size=32)
    )
    # Cross-entropy-dominant JEM: a light CD term (cd_weight=0.1) keeps the
    # energy a meaningful density while cross-entropy does the classification.
    # conditional_negatives=False — we detect OOD by energy, not generate.
    # (A heavier CD term was tried and did not improve near-OOD AUROC at this
    # budget — it only slowed classification.)
    sampler = ebm.LangevinDynamics(
        step_size=10.0, noise_scale=0.005, grad_clip=0.01, steps=40, clamp=(-1, 1)
    )
    loss_fn = ebm.JEMLoss(
        ebm.ContrastiveDivergence(
            sampler,
            buffer=ebm.ReplayBuffer(10_000, (3, 32, 32), reinit_prob=0.05, init_fn=uniform_init),
            energy_reg=1.0,
        ),
        cd_weight=0.1,
    )

    trainer = ebm.Trainer(energy, loss_fn, lr=1e-4, ema_decay=0.999)
    print(f"training on {trainer.device} for {steps} steps")
    trainer.fit((x, y), steps=steps, batch_size=128)

    model = trainer.ema.module if trainer.ema is not None else energy
    model.eval()
    device = trainer.device

    x_test, y_test = ebm.datasets.cifar10(train=False, return_labels=True)
    with torch.no_grad():
        preds = torch.cat(
            [model.logits(chunk.to(device)).argmax(1).cpu() for chunk in x_test.split(1000)]
        )
    acc = (preds == y_test).float().mean().item()

    x_ood = ebm.datasets.cifar100(train=False)  # never seen during training
    in_batch, ood_batch = x_test[:2000], x_ood[:2000]
    auroc = ebm.eval.ood_auroc(lambda t: model(t.to(device)), in_batch, ood_batch)
    print(f"CIFAR-10 test accuracy: {acc:.4f}   CIFAR-10-vs-CIFAR-100 OOD AUROC: {auroc:.4f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ebm.viz.show_images(x_test[:16], nrow=8, ax=axes[0], title="CIFAR-10 test images")
    ebm.viz.energy_histogram(
        lambda t: model(t.to(device)),
        {"CIFAR-10 (in)": in_batch, "CIFAR-100 (OOD)": ood_batch},
        ax=axes[1],
    )
    axes[1].set_title(f"energy by dataset\naccuracy {acc:.3f}, OOD AUROC {auroc:.3f}")
    out = Path(__file__).parent / "cifar_ood_result.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
