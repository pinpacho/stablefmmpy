# Example Notebooks

Each notebook is self-contained and imports directly from `stablefmmpy`.
Install the package first (`pip install -e ..` from this directory) then open
any notebook with Jupyter.

---

## 01 — Benchmark Tables

**`01_benchmarks.ipynb`**  
Reproduces the benchmark tables from the core papers using `BenchmarkSuite`.

- **[M2D] Table 6.1** — Cauchy kernel (scale = 1e-4): balanced vs. naive error for
  r = 10…100.  Naive overflows to Inf for r ≥ 60; balanced stays at ε_mac.
- **[M2D] Table 6.2** — Logarithmic kernel (scale = 100): same comparison.
- **[HK] Table 6.1** — Helmholtz kernel (k = 100): ‖B‖_max factor stays ≤ 1 for
  balanced at all r; naive diverges from r = 120 onward.
- Combined plot: error vs. r for all kernels, balanced vs. naive.

**Classes used:** `PointSet`, `HelmholtzKernel`, `LeafMatrices`, `ScalingFactors`,
`BenchmarkSuite`

---

## 02 — FMM Solver and Tree Structure

**`02_fmm_solver.ipynb`**  
End-to-end walkthrough of the leaf-only wideband FMM solver.

- Builds a `QuadTree` for two separated clusters and inspects per-level node counts,
  `interaction_list` sizes, and `near_list` sizes.
- Visualises the adaptive quad-tree partition (`QuadTree.visualize`).
- Runs `FMMSolver.solve` and compares to the exact kernel matrix-vector product.
- Sweeps expansion order r and plots relative error (semilogy), showing exponential
  convergence to ε_mac.

**Classes used:** `PointSet`, `HelmholtzKernel`, `FMMSolver`, `QuadTree`, `FMMNode`

---

## 03 — Stability Analysis

**`03_stability_analysis.ipynb`**  
Verifies the norm bounds from [HK] Theorem 2.7.

- `StabilityAnalyzer.verify_norm_bounds`: checks ‖U‖_max ≤ 1 for balanced
  factorisation and reports the full bound dictionary.
- Rank sweep: balanced vs. naive error side by side; shows exponential convergence
  of balanced and catastrophic divergence of naive.
- `StabilityAnalyzer.find_overflow_threshold`: determines the expansion order r at
  which the naive factorisation first produces Inf entries.
- Regime table: `LeafMatrices.regime(delta)` for varying δ; histogram of balanced
  vs. naive ‖U[i,:]‖ row norms.
- `ScalingFactors` log-scale computation walkthrough.

**Classes used:** `PointSet`, `ScalingFactors`, `HelmholtzKernel`, `LeafMatrices`,
`StabilityAnalyzer`

---

## 04 — Backward Error Analysis

**`04_backward_error.ipynb`**  
Quantifies the backward stability of the wideband FMM as a function of problem size.

- Backward error ‖δφ‖/‖φ‖ vs. N for balanced FMM (leaf-only solver) at fixed r.
- Verifies the logarithmic growth bound from [M2D] Theorem 5.1:
  error ≤ l · C(r) · ε_mac where l = O(log N).
- Comparison against the naive (dense) matrix-vector product scaling O(N · ε_mac).
- Rank-vs-error curves showing exponential convergence and the staircase pattern.

**Classes used:** `PointSet`, `HelmholtzKernel`, `FMMSolver`, `QuadTree`,
`StabilityAnalyzer`, `BenchmarkSuite`

---

## 05 — Multi-Level FMM

**`05_multilevel.ipynb`**  
Demonstrates `FMMSolver(multilevel=True)`, the true multi-level FMM implementing
[M2D]/[HK] Algorithm 4.1.

**Theory:** four-pass algorithm — M2M upward (postorder), M2L at every level,
L2L downward (preorder), leaf evaluation + P2P.  Inter-level transitions use the
restriction matrix R built by `LeafMatrices.build_R_lf`; M2M applies R directly,
L2L applies R.T (plain transpose).

**Contents:**

1. **Tree structure** — per-level node counts and interaction-list sizes for
   `multilevel=True` vs. `multilevel=False`; quad-tree visualisation.
2. **Validation** — leaf-only vs. multi-level error vs. r table and plot.
3. **[M2D] Tables 6.1 & 6.2** — Cauchy and logarithmic kernel backward error
   (single-leaf balanced UBV, balanced vs. naive).
4. **[HK] Table 6.1** — Helmholtz (k = 100) ‖B‖_max and error table;
   backward error vs. tree depth showing O(l · ε_mac) bound.
5. **Error vs. N sweep** — logarithmic growth of error with N confirmed on loglog plot.
6. **Combined backward error** — all three kernels (Cauchy, Log, Helmholtz) in one
   figure, plus depth-sweep and N-sweep plots verifying [M2D] Theorem 5.1 for the
   multi-level solver.

**Classes used:** all 9 — `PointSet`, `ScalingFactors`, `HelmholtzKernel`,
`LeafMatrices`, `BenchmarkSuite`, `FMMNode`, `QuadTree`, `FMMSolver`,
`StabilityAnalyzer`

---

## Regenerating Notebooks

Each notebook has a companion generator script:

```bash
python gen_01_benchmarks.py        # writes 01_benchmarks.ipynb
python gen_02_fmm_solver.py        # writes 02_fmm_solver.ipynb
python gen_03_stability_analysis.py
python gen_04_backward_error.py
python gen_05_multilevel.py        # writes 05_multilevel.ipynb
```

Run from the `examples/` directory (or `stablefmmpy/examples/` from the repo root).
