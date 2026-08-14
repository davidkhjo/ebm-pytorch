"""Benchmark every training method on two-moons with the library's own eval stack.

For each loss family: train on the same data, draw 4000 samples, then score
with the Fréchet distance, an AIS-bracketed test log-likelihood, and wall
time. Prints a markdown table and writes benchmark_result.png.

Run:  python examples/benchmark_losses.py   (CPU, ~15 minutes)
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

import ebm

N_EVAL = 4000
SEED = 0


def langevin_generate(energy, n):
    sampler = ebm.LangevinDynamics(step_size=1e-2, steps=500)
    return sampler.sample(energy, torch.randn(n, 2))


def at_sigma(net, sigma):
    """Adapt a conditional energy to a plain one at a fixed noise level."""
    return lambda z: net(z, torch.full((len(z),), sigma))


def bench_cd(train):
    energy = ebm.nets.MLPEnergy(dim=2, hidden=(128, 128))
    loss_fn = ebm.ContrastiveDivergence(
        ebm.LangevinDynamics(step_size=1e-2, steps=60), energy_reg=0.1
    )
    ebm.Trainer(energy, loss_fn, lr=1e-3, device="cpu").fit(train, steps=6000, batch_size=256)
    return energy, langevin_generate(energy, N_EVAL)


def bench_pcd(train):
    energy = ebm.nets.MLPEnergy(dim=2, hidden=(128, 128))
    loss_fn = ebm.ContrastiveDivergence(
        ebm.LangevinDynamics(step_size=1e-2, steps=60),
        buffer=ebm.ReplayBuffer(8192, (2,)),
        energy_reg=0.1,
    )
    ebm.Trainer(energy, loss_fn, lr=1e-3, device="cpu").fit(train, steps=6000, batch_size=256)
    return energy, langevin_generate(energy, N_EVAL)


def bench_nce(train):
    energy = ebm.nets.MLPEnergy(dim=2, hidden=(128, 128))
    loss_fn = ebm.NoiseContrastiveEstimation(noise_ratio=4)
    ebm.Trainer(energy, loss_fn, lr=1e-3, device="cpu").fit(train, steps=6000, batch_size=256)
    return energy, langevin_generate(energy, N_EVAL)


def bench_ssm(train):
    energy = ebm.nets.MLPEnergy(dim=2, hidden=(128, 128))
    loss_fn = ebm.SlicedScoreMatching()
    ebm.Trainer(energy, loss_fn, lr=1e-3, device="cpu").fit(train, steps=6000, batch_size=256)
    return energy, langevin_generate(energy, N_EVAL)


NCSN_SIGMAS = ebm.geometric_sigmas(2.0, 0.05, 12)


def bench_ncsn(train):
    net = ebm.nets.NoiseConditionalMLPEnergy(dim=2, hidden=(128, 128))
    loss_fn = ebm.MultiSigmaDenoisingScoreMatching(NCSN_SIGMAS)
    ebm.Trainer(net, loss_fn, lr=1e-3, device="cpu").fit(train, steps=6000, batch_size=256)
    sampler = ebm.AnnealedLangevinDynamics(NCSN_SIGMAS, step_size=5e-3, steps_per_sigma=100)
    samples = sampler.sample(net, torch.randn(N_EVAL, 2))
    return at_sigma(net, float(NCSN_SIGMAS[-1])), samples


def bench_drl(train):
    net = ebm.nets.NoiseConditionalMLPEnergy(dim=2, hidden=(128, 128))
    loss_fn = ebm.DiffusionRecoveryLikelihood(NCSN_SIGMAS, mcmc_steps=30)
    ebm.Trainer(net, loss_fn, lr=1e-3, device="cpu").fit(train, steps=6000, batch_size=256)
    samples = ebm.drl_sample(net, NCSN_SIGMAS, N_EVAL, (2,), mcmc_steps=50)
    return at_sigma(net, float(NCSN_SIGMAS[-1])), samples


METHODS = {
    "CD (short-run)": bench_cd,
    "Persistent CD": bench_pcd,
    "NCE": bench_nce,
    "Sliced SM": bench_ssm,
    "NCSN (multi-σ DSM)": bench_ncsn,
    "DRL": bench_drl,
}


def evaluate(energy, samples, test):
    fd = ebm.eval.frechet_distance(test, samples)
    # MMD at a structure-scale bandwidth catches blur that FD's two moments miss.
    mmd = ebm.eval.mmd(test[:2000], samples[:2000], bandwidth=0.3)
    lower = ebm.ais_log_z(energy, (2,), base_scale=2.0, n_temps=300, n_chains=256)
    upper = ebm.reverse_ais_log_z(energy, samples[:256], base_scale=2.0, n_temps=300)
    # log p = -E - log Z: the log-Z bracket flips into a log-likelihood bracket.
    ll_hi = ebm.log_likelihood(energy, test, lower.log_z).mean().item()
    ll_lo = ebm.log_likelihood(energy, test, upper.log_z).mean().item()
    # Near convergence the two AIS estimates can cross within noise; report sorted.
    ll_lo, ll_hi = sorted((ll_lo, ll_hi))
    return fd, mmd, ll_lo, ll_hi


def main() -> None:
    torch.manual_seed(SEED)
    train = ebm.datasets.two_moons(8192)
    test = ebm.datasets.two_moons(N_EVAL, generator=torch.Generator().manual_seed(1))

    results = {}
    for name, bench in METHODS.items():
        torch.manual_seed(SEED)
        t0 = time.perf_counter()
        energy, samples = bench(train)
        seconds = time.perf_counter() - t0
        fd, mmd, ll_lo, ll_hi = evaluate(energy, samples, test)
        results[name] = (samples, fd, mmd, ll_lo, ll_hi, seconds)
        print(
            f"done: {name}  fd={fd:.3f}  mmd={mmd:.4f}  "
            f"ll=[{ll_lo:.2f}, {ll_hi:.2f}]  {seconds:.0f}s"
        )

    print("\n| Method | FD ↓ | MMD² (bw 0.3) ↓ | Test log-lik (nats) | Train time |")
    print("|---|---|---|---|---|")
    for name, (_, fd, mmd, ll_lo, ll_hi, seconds) in results.items():
        print(f"| {name} | {fd:.3f} | {mmd:.4f} | [{ll_lo:.2f}, {ll_hi:.2f}] | {seconds:.0f} s |")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharex=True, sharey=True)
    for ax, (name, (samples, _fd, mmd, *_)) in zip(axes.flat, results.items(), strict=True):
        ax.scatter(test[:, 0], test[:, 1], s=3, alpha=0.25, color="#b0b0b0", linewidths=0)
        ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.35, color="#5c50c9", linewidths=0)
        ax.set_title(f"{name}  (MMD² {mmd:.4f})", fontsize=11)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2, 2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#d0d0d0")
    fig.suptitle("Samples by training method — gray: data, purple: model", fontsize=13)
    fig.tight_layout()
    out = Path(__file__).parent / "benchmark_result.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
