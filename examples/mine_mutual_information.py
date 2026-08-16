"""Estimate mutual information with MINE, checked against the Gaussian closed form.

For jointly-Gaussian (X, Y) with correlation rho, the mutual information is exactly
-½ log(1 - rho²). We sweep rho, estimate MI from samples alone with MINE (a small
statistics network maximizing the Donsker-Varadhan bound — no density, no bins),
and plot the estimate against the closed form.

Run:  python examples/mine_mutual_information.py
Outputs mine_result.png next to this script (needs the [viz] extra).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

import ebm


def main() -> None:
    torch.manual_seed(0)
    rhos = [0.0, 0.3, 0.5, 0.7, 0.85, 0.95]
    n = 4000

    estimates, truths = [], []
    for rho in rhos:
        x = torch.randn(n, 1)
        y = rho * x + math.sqrt(1 - rho**2) * torch.randn(n, 1)
        mi = ebm.eval.mutual_information(x, y, epochs=600)
        true = -0.5 * math.log(1 - rho**2) if rho < 1 else float("inf")
        estimates.append(mi)
        truths.append(true)
        print(f"rho={rho:.2f}  MINE={mi:.3f}  true={true:.3f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rhos, truths, "-", color="#b0b0b0", label="−½ log(1−ρ²)")
    ax.scatter(rhos, estimates, s=60, color="#5c50c9", zorder=3, label="MINE estimate")
    ax.set_xlabel("correlation ρ")
    ax.set_ylabel("mutual information (nats)")
    ax.set_title("MINE recovers Gaussian mutual information")
    ax.legend()
    out = Path(__file__).parent / "mine_result.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
