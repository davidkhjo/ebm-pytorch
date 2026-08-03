# ebm-pytorch

A small, reliable PyTorch library for training and using **energy-based models**
(EBMs).

An EBM defines an unnormalized density \(p(x) \propto e^{-E(x)}\) through a
neural network `E: (B, *shape) -> (B,)`. This library provides the pieces that
every EBM project rebuilds from scratch — MCMC samplers, training losses,
replay buffers, diagnostics — as small composable objects with tested, correct
defaults.

!!! note "Name collision"
    Not to be confused with *Explainable Boosting Machines* (interpretml),
    which also go by "EBM". This is the deep-learning kind: LeCun et al.
    (2006), Du & Mordatch (2019), Song & Kingma (2021).

## Install

```bash
pip install ebm-pytorch          # runtime dependency is just torch>=2.0
pip install "ebm-pytorch[viz]"   # + matplotlib plotting helpers
```

## Quickstart

```python
import torch, ebm

energy = ebm.nets.MLPEnergy(dim=2, hidden=(128, 128))
sampler = ebm.LangevinDynamics(step_size=1e-2, steps=60)
loss_fn = ebm.ContrastiveDivergence(sampler, buffer=ebm.ReplayBuffer(8192, (2,)))

trainer = ebm.Trainer(energy, loss_fn, lr=1e-3)
trainer.fit(ebm.datasets.two_moons(8192), steps=3000, batch_size=256)

samples = sampler.sample(energy, torch.randn(2000, 2), steps=500)
```

The `Trainer` is optional sugar — the underlying loop is plain PyTorch:

```python
opt = torch.optim.Adam(energy.parameters(), lr=1e-3)
for step in range(3000):
    x = data[torch.randint(len(data), (256,))]
    out = loss_fn(energy, x)  # LossOutput(loss, metrics, x_neg)
    opt.zero_grad()
    out.loss.backward()
    opt.step()
```

## Conventions that matter

- **Sign:** \(p \propto e^{-E}\) — low energy is high probability, everywhere,
  with no exceptions. Samplers *descend* the energy gradient; training pushes
  data energy *down*.
- **Energy functions are plain callables** `(B, *event_shape) -> (B,)`. Any
  function or `nn.Module` with that signature works with every sampler, loss,
  and evaluation tool in the library. Noise-conditional energies take
  `(x, sigma)` with `sigma` of shape `(B,)`.
- **Stop-gradients:** MCMC negatives are detached and the energy network's
  parameters are frozen during sampling; parameter gradients flow only through
  `E(x_data) - E(x_neg)`. Score-matching losses do the opposite and backprop
  through the score (`create_graph=True`). The library enforces this so you
  cannot get it silently wrong.
- **Losses are `nn.Module`s** returning `LossOutput(loss, metrics, x_neg)`;
  supervised losses (like `JEMLoss`) are called as `loss_fn(energy, x, y)`.

## Where to go next

- [Training methods](training.md) — which loss to use and the recipes that
  make each one converge.
- [Sampling](sampling.md) — the sampler catalog and how to tune each one.
- [Evaluating EBMs](evaluation.md) — honest log-likelihoods via log-Z
  bracketing, FID, OOD detection.
- [Composing energies](composition.md) — products, mixtures, tempering, and
  class-conditional energies.
- [Examples](examples.md) — full training walkthroughs with figures.
