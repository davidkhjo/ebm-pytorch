"""Ready-made energy networks.

Design choices follow standard EBM practice: smooth activations (SiLU/Swish),
no batch normalization (it breaks per-sample energies and MCMC), and optional
spectral normalization for Lipschitz control (Du & Mordatch, 2019).
"""

from ebm.nets.flow import AffineCouplingFlow
from ebm.nets.lattice import RBM, IsingEnergy, PottsEnergy
from ebm.nets.mlp_conv import (
    ConvClassifier,
    ConvEnergy,
    MLPEnergy,
    ResNetEnergy,
)
from ebm.nets.mlp_conv import _ResBlock as _ResBlock  # re-export for tests
from ebm.nets.noise_conditional import (
    NoiseConditionalConvEnergy,
    NoiseConditionalMLPEnergy,
)
from ebm.nets.noise_conditional import (
    _GaussianFourierFeatures as _GaussianFourierFeatures,  # re-export for tests
)
from ebm.nets.targets import BananaEnergy, FunnelEnergy, GaussianMixtureEnergy

__all__ = [
    "MLPEnergy",
    "ConvClassifier",
    "ConvEnergy",
    "ResNetEnergy",
    "NoiseConditionalMLPEnergy",
    "NoiseConditionalConvEnergy",
    "IsingEnergy",
    "PottsEnergy",
    "RBM",
    "FunnelEnergy",
    "GaussianMixtureEnergy",
    "BananaEnergy",
    "AffineCouplingFlow",
]
