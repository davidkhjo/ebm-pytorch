"""Evaluation helpers.

Relative diagnostics (OOD scores, batched energies, Fréchet distance / FID,
goodness-of-fit) plus absolute log-likelihood via annealed importance sampling
and the probability-flow ODE.
"""

from ebm.ais import AISResult, ais_log_z, log_likelihood, reverse_ais_log_z
from ebm.eval.diagnostics import (
    effective_sample_size,
    energies,
    ood_auroc,
    split_rhat,
)
from ebm.eval.goodness_of_fit import (
    classifier_two_sample_test,
    fisher_divergence,
    kernel_stein_discrepancy,
    mutual_information,
)
from ebm.eval.likelihood import bits_per_dim, pf_ode_log_likelihood
from ebm.eval.sample_quality import (
    frechet_distance,
    inception_score,
    mmd,
    precision_recall,
)

__all__ = [
    "energies",
    "frechet_distance",
    "mmd",
    "ood_auroc",
    "ais_log_z",
    "reverse_ais_log_z",
    "log_likelihood",
    "bits_per_dim",
    "effective_sample_size",
    "split_rhat",
    "precision_recall",
    "inception_score",
    "kernel_stein_discrepancy",
    "classifier_two_sample_test",
    "fisher_divergence",
    "mutual_information",
    "pf_ode_log_likelihood",
    "AISResult",
]
