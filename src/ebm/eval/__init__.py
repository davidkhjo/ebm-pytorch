"""Evaluation helpers.

Relative diagnostics (OOD scores, batched energies, Fréchet distance / FID,
goodness-of-fit) plus absolute log-likelihood via annealed importance sampling
and the probability-flow ODE.
"""

from ebm.ais import AISResult, ais_log_z, log_likelihood, reverse_ais_log_z
from ebm.eval.calibration import (
    expected_calibration_error,
    reliability_curve,
    temperature_scale,
)
from ebm.eval.diagnostics import (
    autocorrelation,
    effective_sample_size,
    energies,
    ensemble_disagreement,
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
    "AISResult",
    "ais_log_z",
    "autocorrelation",
    "bits_per_dim",
    "classifier_two_sample_test",
    "effective_sample_size",
    "energies",
    "ensemble_disagreement",
    "expected_calibration_error",
    "fisher_divergence",
    "frechet_distance",
    "inception_score",
    "kernel_stein_discrepancy",
    "log_likelihood",
    "mmd",
    "mutual_information",
    "ood_auroc",
    "pf_ode_log_likelihood",
    "precision_recall",
    "reliability_curve",
    "reverse_ais_log_z",
    "split_rhat",
    "temperature_scale",
]
