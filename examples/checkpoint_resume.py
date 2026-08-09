"""Checkpoint a training run and resume it — Trainer.save / Trainer.load.

Image-scale runs take minutes; an interruption should not cost the whole run.
`Trainer.save(path)` captures everything needed to continue — energy weights,
optimizer moments, the EMA copy, any loss parameters, the PCD replay buffer,
and the step counter — and `Trainer.load(path)` restores it into a matching
trainer. Periodic checkpointing pairs naturally with the `callback` hook.

This demo trains a small 2D EBM, checkpoints every 200 steps from the callback,
then rebuilds a fresh trainer, loads the checkpoint, verifies the energy and
step counter carried over, and continues training. Runs on CPU in seconds.

Run:  python examples/checkpoint_resume.py
"""

from __future__ import annotations

from pathlib import Path

import torch

import ebm


def build_trainer():
    """A fresh trainer with the SAME architecture — required before load()."""
    energy = ebm.nets.MLPEnergy(dim=2, hidden=(64, 64))
    loss_fn = ebm.ContrastiveDivergence(
        ebm.LangevinDynamics(step_size=0.01, steps=40),
        buffer=ebm.ReplayBuffer(2048, (2,)),
        energy_reg=0.1,
    )
    return ebm.Trainer(energy, loss_fn, lr=1e-3, ema_decay=0.99, device="cpu")


def main() -> None:
    torch.manual_seed(0)
    data = ebm.datasets.two_moons(4096)
    ckpt = Path(__file__).parent / "checkpoint.pt"

    # --- run 1: train with periodic checkpointing via the callback hook ---
    trainer = build_trainer()
    trainer.callback = lambda step, out: trainer.save(ckpt) if step % 200 == 0 else None
    trainer.fit(data, steps=600, batch_size=256, verbose=False)
    print(f"run 1 trained to step {trainer.step_count}; last checkpoint -> {ckpt.name}")

    # --- run 2: a fresh process rebuilds and resumes from the checkpoint ---
    resumed = build_trainer()
    x = data[:64]
    assert not torch.allclose(resumed.energy(x), trainer.energy(x)), "should differ pre-load"

    resumed.load(ckpt)
    print(f"run 2 resumed at step {resumed.step_count}")
    assert resumed.step_count == trainer.step_count
    assert torch.allclose(resumed.energy(x), trainer.energy(x), atol=1e-6)
    assert torch.allclose(resumed.ema.module(x), trainer.ema.module(x), atol=1e-6)
    assert torch.equal(resumed.loss_fn.buffer.data, trainer.loss_fn.buffer.data)
    print("energy, EMA, and replay buffer all restored exactly")

    resumed.fit(data, steps=400, batch_size=256, verbose=False)
    print(f"run 2 continued to step {resumed.step_count} (600 + 400)")

    ckpt.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
