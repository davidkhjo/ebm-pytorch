import torch
from torch import nn

import ebm


def small_classifier(k=3):
    return nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, k))


def test_classifier_energy_identities():
    energy = ebm.ClassifierEnergy(small_classifier(k=3))
    x = torch.randn(16, 2)
    logits = energy.logits(x)
    assert torch.allclose(energy(x), -logits.logsumexp(dim=1))
    y = torch.randint(3, (16,))
    assert torch.allclose(energy.conditional(x, y), -logits.gather(1, y.reshape(-1, 1)).squeeze(1))
    assert torch.allclose(energy.conditional(x, 1), -logits[:, 1])
    # logsumexp >= max logit, so marginal energy <= any conditional energy.
    assert (energy(x) <= energy.conditional(x, y) + 1e-5).all()


def test_conditional_energy_sampler_compatible():
    energy = ebm.ClassifierEnergy(small_classifier())
    sampler = ebm.LangevinDynamics(step_size=0.01, steps=5)
    with torch.no_grad():
        samples = sampler.sample(energy.condition(0), torch.randn(8, 2))
    assert samples.shape == (8, 2)
    assert all(p.requires_grad for p in energy.parameters())


def test_conv_classifier_as_image_jem():
    net = ebm.nets.ConvClassifier(num_classes=10, in_channels=1, image_size=28, channels=(8, 16))
    energy = ebm.ClassifierEnergy(net)
    x = torch.randn(4, 1, 28, 28)
    assert energy.logits(x).shape == (4, 10)
    assert energy(x).shape == (4,)
    # Class-conditional sampling works on image tensors.
    sampler = ebm.LangevinDynamics(step_size=0.1, steps=3, clamp=(-1, 1))
    samples = sampler.sample(energy.condition(7), torch.rand(4, 1, 28, 28) * 2 - 1)
    assert samples.shape == (4, 1, 28, 28)
    assert all(p.requires_grad for p in energy.parameters())


def test_jem_loss_grads_and_metrics():
    energy = ebm.ClassifierEnergy(small_classifier())
    loss_fn = ebm.JEMLoss(ebm.ContrastiveDivergence(ebm.LangevinDynamics(0.05, steps=5)))
    x, y = torch.randn(32, 2), torch.randint(3, (32,))
    out = loss_fn(energy, x, y)
    assert out.x_neg is not None and not out.x_neg.requires_grad
    out.loss.backward()
    assert all(p.grad is not None for p in energy.parameters())
    assert {"ce", "acc", "cd_loss", "energy_gap"} <= out.metrics.keys()


def test_trainer_supervised_and_regressions():
    x, y = torch.randn(256, 2), torch.randint(2, (256,))

    # (X, Y) tensor pair with a supervised loss.
    energy = ebm.ClassifierEnergy(small_classifier(k=2))
    loss_fn = ebm.JEMLoss(ebm.ContrastiveDivergence(ebm.LangevinDynamics(0.05, steps=3)))
    trainer = ebm.Trainer(energy, loss_fn, device="cpu")
    assert len(trainer.fit((x, y), steps=5, verbose=False)) == 5

    # Labeled DataLoader with an unsupervised loss: labels silently dropped.
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=32)
    mlp = ebm.nets.MLPEnergy(dim=2, hidden=(8,))
    trainer = ebm.Trainer(mlp, ebm.DenoisingScoreMatching(sigma=0.3), device="cpu")
    assert len(trainer.fit(loader, steps=5, verbose=False)) == 5

    # Supervised loss without labels must fail loudly.
    trainer = ebm.Trainer(energy, loss_fn, device="cpu")
    try:
        trainer.fit(x, steps=1, verbose=False)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for missing labels")


def test_jem_conditional_negatives_train_conditional_sampling():
    # Two well-separated Gaussians; with conditional_negatives the per-class
    # sampling paths are trained, so condition(k) generation lands correctly.
    n = 800
    x0 = torch.randn(n, 2) * 0.5 + torch.tensor([-2.0, 0.0])
    x1 = torch.randn(n, 2) * 0.5 + torch.tensor([2.0, 0.0])
    x = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])

    energy = ebm.ClassifierEnergy(small_classifier(k=2))
    loss_fn = ebm.JEMLoss(
        ebm.ContrastiveDivergence(
            ebm.LangevinDynamics(step_size=0.05, steps=20),
            buffer=ebm.ReplayBuffer(1024, (2,)),
            energy_reg=0.05,
        ),
        conditional_negatives=True,
    )
    trainer = ebm.Trainer(energy, loss_fn, lr=1e-3, device="cpu")
    trainer.fit((x, y), steps=300, batch_size=128, verbose=False)

    sampler = ebm.LangevinDynamics(step_size=0.05, steps=100)
    s0 = sampler.sample(energy.condition(0), torch.randn(128, 2))
    s1 = sampler.sample(energy.condition(1), torch.randn(128, 2))
    assert s0[:, 0].mean() < -1
    assert s1[:, 0].mean() > 1
    # Buffer is still used by the conditional path.
    assert loss_fn.cd.buffer._last_idx is not None


def test_jem_joint_training_two_gaussians():
    n = 1000
    x0 = torch.randn(n, 2) * 0.5 + torch.tensor([-2.0, 0.0])
    x1 = torch.randn(n, 2) * 0.5 + torch.tensor([2.0, 0.0])
    x = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])

    energy = ebm.ClassifierEnergy(
        nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))
    )
    loss_fn = ebm.JEMLoss(
        ebm.ContrastiveDivergence(
            ebm.LangevinDynamics(step_size=0.05, steps=20),
            buffer=ebm.ReplayBuffer(1024, (2,)),
            energy_reg=0.05,
        )
    )
    trainer = ebm.Trainer(energy, loss_fn, lr=1e-3, device="cpu")
    trainer.fit((x, y), steps=300, batch_size=128, verbose=False)

    with torch.no_grad():
        acc = (energy.logits(x).argmax(dim=1) == y).float().mean().item()
        e_data = energy(x[:512]).mean()
        e_far = energy(x[:512] + torch.tensor([0.0, 8.0])).mean()
    assert acc > 0.9
    assert e_data < e_far

    # Conditional sampling lands on the right side of the x-axis per class.
    sampler = ebm.LangevinDynamics(step_size=0.05, steps=100)
    s0 = sampler.sample(energy.condition(0), torch.randn(256, 2))
    s1 = sampler.sample(energy.condition(1), torch.randn(256, 2))
    assert s0[:, 0].mean() < 0
    assert s1[:, 0].mean() > 0
