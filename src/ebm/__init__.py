"""ebm: train and use energy-based models in PyTorch.

Convention everywhere: ``p(x) ∝ exp(-E(x))`` — low energy is high probability.
An energy function is any callable ``(B, *event_shape) -> (B,)``.
"""

from ebm import datasets, eval, nets, viz
from ebm.ais import AISResult, ais_log_z, log_likelihood, reverse_ais_log_z
from ebm.buffer import ReplayBuffer
from ebm.compose import MixtureEnergy, SumEnergy, TemperedEnergy
from ebm.diffusion import DDPMAncestralSampler, VPDenoisingScoreMatching, VPSchedule
from ebm.energy import ConditionalEnergyFn, EnergyFn, EnergyModel, score
from ebm.jem import ClassifierEnergy, ConditionalEnergy, GuidedEnergy, JEMLoss
from ebm.losses import (
    ConcreteScoreMatching,
    ContrastiveDivergence,
    DenoisingScoreMatching,
    DiffusionRecoveryLikelihood,
    EnergyDiscrepancy,
    ExactScoreMatching,
    LossOutput,
    MultiSigmaDenoisingScoreMatching,
    NoiseContrastiveEstimation,
    PseudoLikelihood,
    RatioMatching,
    SlicedScoreMatching,
    drl_sample,
    geometric_sigmas,
)
from ebm.samplers import (
    HMC,
    MALA,
    SVGD,
    AdaptiveMALA,
    AnnealedLangevinDynamics,
    CategoricalGibbsWithGradients,
    GibbsSampler,
    GibbsWithGradients,
    LangevinDynamics,
    ParallelTempering,
    PreconditionedLangevin,
    PredictorCorrector,
    ProbabilityFlowODE,
    Sampler,
    TemperedTransitions,
    UnderdampedLangevin,
)
from ebm.training import Trainer
from ebm.utils import EMA

__version__ = "0.15.0"

__all__ = [
    "EMA",
    "HMC",
    "MALA",
    "SVGD",
    "AISResult",
    "AdaptiveMALA",
    "AnnealedLangevinDynamics",
    "CategoricalGibbsWithGradients",
    "ClassifierEnergy",
    "ConcreteScoreMatching",
    "ConditionalEnergy",
    "ConditionalEnergyFn",
    "ContrastiveDivergence",
    "DDPMAncestralSampler",
    "DenoisingScoreMatching",
    "DiffusionRecoveryLikelihood",
    "EnergyDiscrepancy",
    "EnergyFn",
    "EnergyModel",
    "ExactScoreMatching",
    "GibbsSampler",
    "GibbsWithGradients",
    "GuidedEnergy",
    "JEMLoss",
    "LangevinDynamics",
    "LossOutput",
    "MixtureEnergy",
    "MultiSigmaDenoisingScoreMatching",
    "NoiseContrastiveEstimation",
    "ParallelTempering",
    "PreconditionedLangevin",
    "PredictorCorrector",
    "ProbabilityFlowODE",
    "PseudoLikelihood",
    "RatioMatching",
    "ReplayBuffer",
    "Sampler",
    "SlicedScoreMatching",
    "SumEnergy",
    "TemperedEnergy",
    "TemperedTransitions",
    "Trainer",
    "UnderdampedLangevin",
    "VPDenoisingScoreMatching",
    "VPSchedule",
    "ais_log_z",
    "datasets",
    "drl_sample",
    "eval",
    "geometric_sigmas",
    "log_likelihood",
    "nets",
    "reverse_ais_log_z",
    "score",
    "viz",
]
