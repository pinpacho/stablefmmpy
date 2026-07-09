# stablefmmpy

A stable Python implementation of the wideband Fast Multipole Method (FMM) for the 2D Helmholtz kernel and related 2D kernels (generalised Cauchy, logarithmic), based on:

> **[HK]** "A Stable Matrix Version of the Wideband FMM for the 2D Helmholtz Kernel"  
> Michelle, Ou, Xia — preprint 2024

> **[M2D]** "A Stable Matrix Version of the 2D Fast Multipole Method"  
> Ou, Michelle, Xia — SIAM J. Matrix Anal. Appl. 46(1), 2025

## Why this library?

The classical FMM is numerically unstable in the low-frequency regime (k·δ ≪ 1) because Hankel function values grow factorially with expansion order r. This library implements the **balanced matrix version** from [HK]/[M2D]: scaling factors λ_{x,p} keep ‖U‖_max ≤ 1, giving backward error that grows only logarithmically with N — instead of exponentially.

| Method | ‖B‖_max at r=180 | Relative error |
|--------|-----------------|----------------|
| Naive FMM | Inf | NaN |
| Balanced FMM (this library) | ≈ 0.629 | ≈ 1e-14 |

## Installation

```bash
cd stablefmmpy
pip install -e .
```

Requires Python 3.12+, numpy, scipy, matplotlib (all standard scientific Python).

## Quick Start

```python
import numpy as np
from stablefmmpy import PointSet, FMMSolver, HelmholtzKernel

rng = np.random.default_rng(42)
X = PointSet.random_uniform(n=100, center=0+0j, radius=0.01, rng=rng)
Y = PointSet.random_uniform(n=100, center=0.1+0j, radius=0.01, rng=rng)
q = rng.standard_normal(100) + 1j * rng.standard_normal(100)

# Fast approximation: O(N * r)
solver = FMMSolver(k=100.0, r=30, tau=0.6, N0=32, balanced=True)
phi = solver.solve(X, Y, q)

# Exact reference: O(N^2)
kern = HelmholtzKernel(100.0)
phi_exact = kern.matvec(X, Y, q)
err = kern.relative_error(phi, phi_exact)
print(f"Relative error: {err:.2e}")   # -> ~1e-14
```

## Running Benchmarks

```python
from stablefmmpy import BenchmarkSuite

bs = BenchmarkSuite()

# Reproduce Table 6.1 from [HK] (Helmholtz, k=100)
rows = bs.run_helmholtz_table61()
for row in rows:
    print(f"r={row['r']:3d}  stable_B={row['stable_B']:.4f}  naive_B={row['naive_B']}")

# Reproduce Table 6.1 from [M2D] (Cauchy kernel)
rows2 = bs.run_multipole2d_table61()
for row in rows2:
    print(f"r={row['r']:3d}  balanced_err={row['balanced_err']:.2e}  naive_err={row['naive_err']:.2e}")
```

## Verifying Stability

```python
from stablefmmpy import PointSet, StabilityAnalyzer

rng = np.random.default_rng(0)
X = PointSet.random_uniform(40, center=0+0j, radius=0.005, rng=rng)
Y = PointSet.random_uniform(40, center=0.05+0j, radius=0.005, rng=rng)

sa = StabilityAnalyzer(k=50.0, r=15)
bounds = sa.verify_norm_bounds(X, Y)
print(bounds)
# -> {'U_max_balanced': 0.9..., 'theorem_satisfied': True, ...}
```

## Module Layout

```
src/stablefmmpy/
├── __init__.py      public API
├── core.py          PointSet, ScalingFactors, HelmholtzKernel
├── matrices.py      LeafMatrices (U, B, V, R builders — LF and HF regimes)
├── tree.py          FMMNode, QuadTree (adaptive hierarchical partitioning)
├── solver.py        FMMSolver (wideband FMM algorithm)
└── analysis.py      BenchmarkSuite, StabilityAnalyzer
```

## Running Tests

```bash
pytest tests/ -v
```

## Mathematical Background

**Problem:** Given point sets X = {xᵢ} and Y = {yⱼ} in ℂ and a charge vector q, evaluate φᵢ = Σⱼ H₀(k|xᵢ − yⱼ|) qⱼ.

**Low-rank factorization:** For well-separated sets (separation ratio τ), the kernel admits K ≈ U B Vᵀ. The balanced version scales U by λ_{x,p} = max{1, p! · (2/(kδ))^p} so that ‖U‖_max ≤ 1 [HK Theorem 2.7].

**Two-regime architecture:**
- **Low-frequency** (k·δ ≤ r/e): Bessel/Hankel recurrences with balancing
- **High-frequency** (k·δ > r/e): equispaced DFT basis (always stable)

## Citation

```bibtex
@article{michelle2024helmholtz,
  title={A Stable Matrix Version of the Wideband {FMM} for the 2D {Helmholtz} Kernel},
  author={Michelle, Ou, Xia},
  year={2024},
  note={preprint}
}

@article{ou2025multipole2d,
  title={A Stable Matrix Version of the 2D Fast Multipole Method},
  author={Ou, Michelle, Xia},
  journal={SIAM Journal on Matrix Analysis and Applications},
  volume={46},
  number={1},
  year={2025}
}
```
