"""ebm: train and use energy-based models in PyTorch.

Convention everywhere: ``p(x) ∝ exp(-E(x))`` — low energy is high probability.
An energy function is any callable ``(B, *event_shape) -> (B,)``.
"""

from ebm import datasets, eval, nets, viz
from ebm.buffer import ReplayBuffer
from ebm.energy import EnergyFn, EnergyModel, score
from ebm.losses import (
    ContrastiveDivergence,
    DenoisingScoreMatching,
    LossOutput,
    NoiseContrastiveEstimation,
    SlicedScoreMatching,
)
from ebm.samplers import HMC, MALA, GibbsWithGradients, LangevinDynamics, Sampler
from ebm.training import Trainer
from ebm.utils import EMA

__version__ = "0.1.0"

__all__ = [
    "EnergyFn",
    "EnergyModel",
    "score",
    "Sampler",
    "LangevinDynamics",
    "MALA",
    "HMC",
    "GibbsWithGradients",
    "ReplayBuffer",
    "LossOutput",
    "ContrastiveDivergence",
    "DenoisingScoreMatching",
    "SlicedScoreMatching",
    "NoiseContrastiveEstimation",
    "Trainer",
    "EMA",
    "datasets",
    "eval",
    "nets",
    "viz",
]
