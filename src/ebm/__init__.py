"""ebm: train and use energy-based models in PyTorch.

Convention everywhere: ``p(x) ∝ exp(-E(x))`` — low energy is high probability.
An energy function is any callable ``(B, *event_shape) -> (B,)``.
"""

from ebm import datasets, eval, nets, viz
from ebm.ais import AISResult, ais_log_z, log_likelihood, reverse_ais_log_z
from ebm.buffer import ReplayBuffer
from ebm.compose import MixtureEnergy, SumEnergy, TemperedEnergy
from ebm.energy import ConditionalEnergyFn, EnergyFn, EnergyModel, score
from ebm.jem import ClassifierEnergy, ConditionalEnergy, JEMLoss
from ebm.losses import (
    ContrastiveDivergence,
    DenoisingScoreMatching,
    DiffusionRecoveryLikelihood,
    LossOutput,
    MultiSigmaDenoisingScoreMatching,
    NoiseContrastiveEstimation,
    SlicedScoreMatching,
    drl_sample,
    geometric_sigmas,
)
from ebm.samplers import (
    HMC,
    MALA,
    AnnealedLangevinDynamics,
    CategoricalGibbsWithGradients,
    GibbsWithGradients,
    LangevinDynamics,
    Sampler,
)
from ebm.training import Trainer
from ebm.utils import EMA

__version__ = "0.8.0"

__all__ = [
    "EnergyFn",
    "ConditionalEnergyFn",
    "EnergyModel",
    "score",
    "ClassifierEnergy",
    "ConditionalEnergy",
    "JEMLoss",
    "SumEnergy",
    "MixtureEnergy",
    "TemperedEnergy",
    "Sampler",
    "LangevinDynamics",
    "MALA",
    "HMC",
    "GibbsWithGradients",
    "CategoricalGibbsWithGradients",
    "AnnealedLangevinDynamics",
    "ReplayBuffer",
    "LossOutput",
    "ContrastiveDivergence",
    "DenoisingScoreMatching",
    "DiffusionRecoveryLikelihood",
    "drl_sample",
    "MultiSigmaDenoisingScoreMatching",
    "SlicedScoreMatching",
    "NoiseContrastiveEstimation",
    "geometric_sigmas",
    "Trainer",
    "EMA",
    "ais_log_z",
    "reverse_ais_log_z",
    "log_likelihood",
    "AISResult",
    "datasets",
    "eval",
    "nets",
    "viz",
]
