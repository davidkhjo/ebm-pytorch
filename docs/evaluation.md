# Evaluating EBMs

Unnormalized models make evaluation the hard part. The library gives you three
levels of rigor.

## Log-likelihood via log-Z bracketing

`ais_log_z` estimates \(\log Z = \log \int e^{-E(x)}\,dx\) by annealed
importance sampling — a stochastic *lower* bound in expectation.
`reverse_ais_log_z` runs the same path backwards from model samples and is a
stochastic *upper* bound. Report both:

```python
lower = ebm.ais_log_z(energy, shape=(2,), n_temps=1000, n_chains=256)
upper = ebm.reverse_ais_log_z(energy, lower.samples, n_temps=1000)

nats = ebm.log_likelihood(energy, x_test, lower.log_z)
bits = ebm.log_likelihood(energy, x_test, lower.log_z, dim=2)  # bits/dim
```

When the two agree, trust the number. When they don't, the gap tells you how
much annealing is missing — increase `n_temps` (many cheap temperatures beat
few long MCMC runs) and check `result.ess`: an effective sample size near 1
means a few chains dominate and the estimate is unreliable.

!!! tip "The reverse bound is only as honest as its samples"
    `reverse_ais_log_z` assumes its inputs are (approximately) model samples —
    use `ais_log_z(...).samples` or a long MCMC run. Poor samples bias the
    estimate further *upward*, so the bracket stays conservative rather than
    falsely tight.

## Sample quality: Fréchet distance / FID

```python
fd = ebm.eval.frechet_distance(x_real, x_generated)             # raw samples
fid = ebm.eval.frechet_distance(x_real, x_gen, feature_fn=inception_pool3)
```

Pure torch (float64, eigendecomposition matrix square root — no scipy).
`feature_fn=None` compares raw flattened samples, right for toy data; passing
an Inception embedding gives standard FID. Use a few thousand samples per side
— small sets bias the covariance term upward.

!!! warning "Fréchet distance only sees two moments"
    A *blurred* version of the data matches mean and covariance almost
    perfectly, so FD barely moves. `ebm.eval.mmd(x, y, bandwidth=...)` —
    unbiased squared MMD with an RBF kernel — sees all moments; pick the
    bandwidth at the scale of the structure you care about (the default
    median heuristic is conservative). The [benchmarks](benchmarks.md) page
    shows a case where FD and MMD genuinely disagree.

## Relative diagnostics

- `ebm.eval.ood_auroc(energy, x_in, x_out)` — AUROC for separating
  in-distribution from OOD data by energy (rank statistic, no sklearn).
- `ebm.eval.energies(energy, x)` — batched energies on CPU, for histograms.
- `ebm.viz.energy_histogram(energy, {"data": x, "model": x_neg})` — the
  single most useful training diagnostic: watch the two histograms merge.
