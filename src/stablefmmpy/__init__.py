"""
stablefmmpy — Stable Wideband Fast Multipole Method for the 2D Helmholtz Kernel.

References:
  [HK]  "A Stable Matrix Version of the Wideband FMM for the 2D Helmholtz Kernel"
        Michelle, Ou, Xia — preprint 2024
  [M2D] "A Stable Matrix Version of the 2D Fast Multipole Method"
        Ou, Michelle, Xia — SIAM J. Matrix Anal. Appl. 46(1), 2025
"""

from .core import PointSet, ScalingFactors, HelmholtzKernel
from .matrices import LeafMatrices
from .tree import FMMNode, QuadTree
from .solver import FMMSolver
from .analysis import StabilityAnalyzer, BenchmarkSuite

__version__ = "0.1.0"

__all__ = [
    "PointSet",
    "ScalingFactors",
    "HelmholtzKernel",
    "LeafMatrices",
    "FMMNode",
    "QuadTree",
    "FMMSolver",
    "StabilityAnalyzer",
    "BenchmarkSuite",
]
