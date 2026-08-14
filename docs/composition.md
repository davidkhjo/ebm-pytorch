# Composing energies

Energies add where densities multiply — composition is where EBMs shine, and
the composition classes are drop-in energies for every sampler, loss, and
evaluation tool.

## Products, mixtures, tempering

```python
product  = ebm.SumEnergy(e1, e2)                    # p ∝ p1 · p2  (product of experts)
weighted = ebm.SumEnergy(e1, e2, weights=[2.0, 1.0])
mixture  = ebm.MixtureEnergy(e1, e2)                # p ∝ (p1 + p2) / 2
sharp    = ebm.TemperedEnergy(e1, temperature=0.5)  # p ∝ p1^2   (T < 1 sharpens)
nested   = ebm.MixtureEnergy(product, sharp)        # compose freely
```

- `SumEnergy` — $E = \sum_i w_i E_i$: the intersection of constraints. Two
  overlapping Gaussians multiply to a tighter Gaussian between them.
- `MixtureEnergy` — $E = -\log \sum_i w_i e^{-E_i}$: the union. Exact
  log-sum-exp, numerically stable.
- `TemperedEnergy` — $E / T$: tempering. A tempered standard Gaussian has
  std $\sqrt{T}$ — useful both for sharpening trained models and for
  building annealing paths.

All three register component `nn.Module`s, so parameters are visible to
optimizers and frozen correctly during sampling. See
[`examples/train_composition.py`](examples.md#composing-energies-product-mixture-tempering)
for a runnable product-of-experts / mixture / tempering demo.

## Class-conditional energies (JEM)

`ClassifierEnergy` gives three energies from one network:

```python
clf = ebm.ClassifierEnergy(net)     # net: (B, *shape) -> (B, K)
clf(x)                              # marginal:   E(x)   = -logsumexp(logits)
clf.conditional(x, y)               # joint:      E(x,y) = -logits[:, y]
cond3 = clf.condition(3)            # E(x | y=3) as a standalone energy module
samples = sampler.sample(cond3, x0) # class-conditional sampling
```

`condition(y)` returns a real `nn.Module` holding the parent, so parameter
freezing works during conditional sampling, and the result composes with
everything above — e.g. `ebm.SumEnergy(clf.condition(3), style_energy)`.
