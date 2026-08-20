"""Self-tuning MALA: no step-size sweep, and a metric for ill-conditioned targets.

Plain MALA needs its ``step_size`` hand-tuned per target: too large and every
proposal is rejected, too small and the chain crawls. ``AdaptiveMALA`` runs a
dual-averaging warmup that drives the acceptance rate to the MALA-optimal 0.574
on its own, then freezes and samples. On an ill-conditioned target (here a
Gaussian with a 100:1 spread between axes) an isotropic step is capped by the
tight direction; ``precondition=True`` learns a diagonal metric that lets the
usable step grow ~7× (3.0 vs 0.43 here) — the same optimal acceptance, but far
faster mixing per step. Both recover the true per-axis std.

Run:  python examples/adaptive_mala.py
Outputs adaptive_mala_result.png next to this script (needs the [viz] extra).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ebm

COV = torch.diag(torch.tensor([25.0, 0.25]))  # 100:1 condition number
PRECISION = torch.linalg.inv(COV)


def energy(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * ((x @ PRECISION) * x).sum(dim=1)


def main() -> None:
    torch.manual_seed(0)
    true_std = COV.diag().sqrt()
    x0 = torch.randn(4000, 2)

    plain = ebm.AdaptiveMALA(step_size=0.1, steps=800, warmup=1000)
    xp = plain.sample(energy, x0.clone())
    print(f"isotropic:      tuned eps={plain.step_size:.3f}  accept={plain.last_accept_rate:.3f}")
    print(f"  per-axis std {xp.std(0).tolist()}  (true {true_std.tolist()})")

    pre = ebm.AdaptiveMALA(step_size=0.1, steps=800, warmup=1000, precondition=True)
    xq = pre.sample(energy, x0.clone())
    m = pre.preconditioner
    print(f"preconditioned: tuned eps={pre.step_size:.3f}  accept={pre.last_accept_rate:.3f}")
    print(f"  learned metric ratio {(m[0] / m[1]).item():.1f}  (true 100)")
    print(f"  per-axis std {xq.std(0).tolist()}  (true {true_std.tolist()})")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, s, title in ((axes[0], xp, "isotropic"), (axes[1], xq, "preconditioned")):
        ax.scatter(s[:, 0], s[:, 1], s=5, alpha=0.3, color="#5c50c9")
        ax.set_title(f"AdaptiveMALA — {title}")
        ax.set_xlim(-16, 16)
        ax.set_ylim(-2, 2)
    fig.suptitle("Self-tuning MALA on a 100:1 ill-conditioned Gaussian")
    out = Path(__file__).parent / "adaptive_mala_result.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
