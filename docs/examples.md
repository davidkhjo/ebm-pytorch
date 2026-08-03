# Examples

Both examples live in [`examples/`](https://github.com/dkjo8/ebm-pytorch/tree/main/examples)
and run on CPU in a few minutes.

## Two moons with persistent CD

`examples/train_two_moons.py` — the canonical smoke test. An `MLPEnergy` is
trained with `ContrastiveDivergence` + `ReplayBuffer` + `energy_reg=0.1`,
then sampled with long-run Langevin:

![Two moons result](assets/two_moons_result.png)

Left to right: data, the learned energy landscape (low energy on the moons),
and fresh samples from noise. The pieces that matter:

```python
energy = ebm.nets.MLPEnergy(dim=2, hidden=(128, 128))
loss_fn = ebm.ContrastiveDivergence(
    ebm.LangevinDynamics(step_size=1e-2, steps=60),
    buffer=ebm.ReplayBuffer(8192, (2,)),
    energy_reg=0.1,       # without this the energies drift to ±500 and diverge
)
ebm.Trainer(energy, loss_fn, lr=1e-3).fit(ebm.datasets.two_moons(8192), steps=6000)
```

## JEM: classify and generate with one network

`examples/train_jem.py` — a 3-layer MLP classifier on labeled two-moons,
trained jointly with `JEMLoss` (cross-entropy + contrastive divergence on the
marginal energy). It reaches 100% accuracy *and* generates each moon on
demand:

![JEM result](assets/jem_result.png)

```python
energy = ebm.ClassifierEnergy(my_2_class_mlp)
loss_fn = ebm.JEMLoss(ebm.ContrastiveDivergence(sampler, buffer=buffer, energy_reg=0.1))
trainer.fit((x, y), steps=4000, batch_size=256)   # supervised batches

moon_0 = sampler.sample(energy.condition(0), noise)  # class-conditional samples
```

The EMA weights (`trainer.ema.module`) give visibly cleaner samples than the
raw weights — use them for anything you show to humans.
