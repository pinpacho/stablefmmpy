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

# Leaf-only solver (default): O(N·r)
solver = FMMSolver(k=10.0, r=20, tau=0.6, N0=32, balanced=True)
phi = solver.solve(X, Y, q)

# True multi-level solver (M2M→M2L@all levels→L2L→P2P): same accuracy
solver_ml = FMMSolver(k=10.0, r=20, tau=0.6, N0=32, balanced=True, multilevel=True)
phi_ml = solver_ml.solve(X, Y, q)

# Exact reference: O(N^2)
kern = HelmholtzKernel(10.0)
phi_exact = kern.matvec(X, Y, q)
print(f"Leaf-only error:   {kern.relative_error(phi, phi_exact):.2e}")    # ~1e-14
print(f"Multi-level error: {kern.relative_error(phi_ml, phi_exact):.2e}") # ~1e-14
```

## Multi-Level FMM

`FMMSolver(multilevel=True)` implements [M2D]/[HK] Algorithm 4.1 — the classical
four-pass FMM traversal:

| Pass | Direction | Operation |
|------|-----------|-----------|
| M2M | Postorder (bottom-up) | `v_parent += R @ v_child` |
| M2L | All levels | `t_node += B @ v_partner`  for each interaction-list partner |
| L2L | Preorder (top-down) | `u_child = R.T @ u_parent + t_child` |
| Eval + P2P | Leaves | `phi += U @ u_leaf + K_exact @ q_near` |

The inter-level restriction matrix R is built by `LeafMatrices.build_R_lf`
([HK §3 Theorem 3.1]):

```
R[p, l] = (λ_parent[|p|] / λ_child[|l|]) * J_{|p-l|}(k|d|) * exp(i(p-l)·arg(d))
```

where `d = child.center − parent.center`.  M2M applies R directly; L2L applies
`R.T` (plain transpose — not conjugate transpose).

`QuadTree.build(multilevel=True)` populates classical FMM interaction lists at every
level (pair A,B added at the coarsest level where they first become well-separated)
plus a reconciliation pass that routes any uncovered leaf pairs to direct P2P.

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
├── __init__.py      public API (9 classes)
├── core.py          PointSet, ScalingFactors, HelmholtzKernel
├── matrices.py      LeafMatrices — U, B, V, R builders (LF and HF regimes)
├── tree.py          FMMNode, QuadTree — adaptive quad-tree, leaf-only and
│                    multilevel interaction lists, M2M/L2L traversal support
├── solver.py        FMMSolver — leaf-only (_solve_wideband) and true
│                    multi-level (_solve_multilevel) wideband FMM
└── analysis.py      BenchmarkSuite, StabilityAnalyzer

tests/
├── test_core.py
├── test_matrices.py
├── test_solver.py
├── test_solver_multilevel.py   ← multi-level FMM correctness + tree structure
└── test_stability.py

examples/               see examples/README.md
├── 01_benchmarks.ipynb
├── 02_fmm_solver.ipynb
├── 03_stability_analysis.ipynb
├── 04_backward_error.ipynb
└── 05_multilevel.ipynb
```

## Running Tests

```bash
pytest tests/ -v
```

All 47 tests pass (38 original + 9 multilevel-specific).

## Mathematical Background

**Problem:** Given point sets X = {xᵢ} and Y = {yⱼ} in ℂ and a charge vector q,
evaluate φᵢ = Σⱼ H₀(k|xᵢ − yⱼ|) qⱼ.  Brute force is O(MN); FMM achieves O(M+N).

**Low-rank factorisation:** For well-separated sets (separation ratio τ), the kernel
admits K ≈ U B Vᵀ.  The balanced version scales U by
λ_{x,p} = max{1, p! · (2/(kδ))^p} so that ‖U‖_max ≤ 1 [HK Theorem 2.7].

**Two-regime architecture:**
- **Low-frequency** (k·δ ≤ r/e): Bessel/Hankel recurrences with balancing
- **High-frequency** (k·δ > r/e): equispaced DFT basis (inherently stable)

**Backward error bound** ([M2D] Theorem 5.1): for a tree of depth l,
‖K − K_approx‖ / ‖K‖ ≤ l · C(r) · ε_mac, where C(r) → 0 exponentially.
Since l = O(log N), the backward error grows logarithmically with N.

## Public API

```python
from stablefmmpy import (
    PointSet,          # point set with bounding disk (center, radius)
    ScalingFactors,    # λ_{x,p} balancing factors
    HelmholtzKernel,   # direct O(MN) evaluator + Cauchy/log factories
    LeafMatrices,      # U, B, V, R matrix builders (LF + HF regimes)
    FMMNode,           # single quad-tree node
    QuadTree,          # adaptive quad-tree (leaf-only or multilevel lists)
    FMMSolver,         # wideband FMM solver (leaf-only or multilevel=True)
    StabilityAnalyzer, # norm-bound verification and rank sweeps
    BenchmarkSuite,    # reproduces Tables 6.1–6.3 from [HK] and [M2D]
)
```

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
