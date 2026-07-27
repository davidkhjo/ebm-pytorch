import torch

import ebm


def test_trainer_learns_low_energy_on_data():
    """End-to-end: short CD training must assign lower energy to data than noise."""
    data = ebm.datasets.eight_gaussians(2000)
    energy = ebm.nets.MLPEnergy(dim=2, hidden=(64, 64))
    buffer = ebm.ReplayBuffer(capacity=1024, shape=(2,))
    sampler = ebm.LangevinDynamics(step_size=0.05, steps=25)
    loss_fn = ebm.ContrastiveDivergence(sampler, buffer=buffer, energy_reg=0.001)

    trainer = ebm.Trainer(energy, loss_fn, lr=1e-3, device="cpu", ema_decay=0.99)
    history = trainer.fit(data, steps=250, batch_size=128, verbose=False)

    assert len(history) == 250
    assert all(torch.isfinite(torch.tensor(h["loss"])) for h in history)

    with torch.no_grad():
        e_data = energy(data[:512]).mean()
        e_noise = energy(4 * torch.randn(512, 2)).mean()
    assert e_data < e_noise

    # EMA copy tracks and is usable as an energy function.
    with torch.no_grad():
        e_ema = trainer.ema.module(data[:16])
    assert e_ema.shape == (16,)


def test_trainer_accepts_dataloader():
    data = torch.utils.data.TensorDataset(torch.randn(64, 2))
    loader = torch.utils.data.DataLoader(data, batch_size=16)
    energy = ebm.nets.MLPEnergy(dim=2, hidden=(8,))
    loss_fn = ebm.DenoisingScoreMatching(sigma=0.3)
    trainer = ebm.Trainer(energy, loss_fn, device="cpu")
    history = trainer.fit(loader, steps=10, verbose=False)
    assert len(history) == 10


def test_trainer_optimizes_loss_parameters():
    energy = ebm.nets.MLPEnergy(dim=2, hidden=(8,))
    loss_fn = ebm.NoiseContrastiveEstimation()
    trainer = ebm.Trainer(energy, loss_fn, device="cpu", lr=0.1)
    trainer.fit(torch.randn(256, 2), steps=20, verbose=False)
    assert loss_fn.log_z.item() != 0.0
