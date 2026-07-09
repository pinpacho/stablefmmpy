"""
analysis.py — BenchmarkSuite, StabilityAnalyzer.

Reproduces Tables 6.1-6.3 from [HK] and Tables 6.1-6.2 from [M2D].

References:
  [HK]  HelmholtzKernel2D.pdf — Michelle, Ou, Xia; preprint 2024
  [M2D] Multipole2D.pdf — Ou, Michelle, Xia; SIAM J. Matrix Anal. Appl. 46(1), 2025
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .core import PointSet, ScalingFactors, HelmholtzKernel
from .matrices import LeafMatrices


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------

class BenchmarkSuite:
    """Reproduces benchmark tables from [HK] and [M2D].

    All tables compare balanced vs naive factorizations by tracking:
    - B_factor = ||B||_max (the balance factor from the paper)
    - rel_error = ||U B V^T q - K q||_2 / ||K q||_2

    Reference benchmark values:
      [HK] Table 6.1: stable B ≈ 0.5287 (constant), naive B → Inf at r=180+
      [HK] Table 6.2: stable B ≈ 0.81 for L̃=2, grows with L̃
      [M2D] Table 6.1: naive error → Inf at r >= 70 (Cauchy, scale=1e-4)
      [M2D] Table 6.2: naive error → Inf at r >= 90 (log, scale=100)
    """

    @staticmethod
    def _make_helmholtz_clusters(N: int, scale: float, seed: int = 42
                                 ) -> Tuple[PointSet, PointSet, np.ndarray]:
        """Return (X, Y, q) for Helmholtz benchmarks."""
        rng = np.random.default_rng(seed)
        r_x = np.sqrt(rng.uniform(0, 1, N))
        th_x = rng.uniform(0, 2 * np.pi, N)
        pts_x = scale * r_x * np.exp(1j * th_x)

        r_y = np.sqrt(rng.uniform(0, 1, N))
        th_y = rng.uniform(0, 2 * np.pi, N)
        pts_y = scale * (6.0 + r_y * np.exp(1j * th_y))

        X = PointSet(pts_x)
        Y = PointSet(pts_y)
        q = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        return X, Y, q

    def run_helmholtz_table61(self, r_values=None, k: float = 100.0,
                              scale: float = 0.0025, N: int = 50,
                              seed: int = 42) -> List[dict]:
        """Reproduce HelmholtzKernel2D Table 6.1.

        k=100, scale=0.0025. Expected: stable B ≈ 0.5287 (constant for all r),
        naive B grows from ~3.5e190 (r=120) to Inf (r>=180). [HK Table 6.1]
        """
        if r_values is None:
            r_values = [120, 130, 140, 150, 160, 170, 180, 190, 200]

        X, Y, q = self._make_helmholtz_clusters(N, scale, seed)
        kern = HelmholtzKernel(k)
        phi_exact = kern.matvec(X, Y, q)

        rows = []
        for r in r_values:
            lm = LeafMatrices(r, k)
            sf_x = ScalingFactors(r, k, X.radius)
            sf_y = ScalingFactors(r, k, Y.radius)

            U_s = lm.build_basis_lf(X, sf_x, balanced=True)
            V_s = lm.build_basis_lf(Y, sf_y, balanced=True)
            B_s = lm.build_B_lf(X, Y, sf_x, sf_y, balanced=True)
            phi_s = U_s @ (B_s @ (V_s.T @ q))
            B_fac_s = lm.balance_factor(B_s)
            err_s = kern.relative_error(phi_s, phi_exact)

            sf0_x = ScalingFactors.__new__(ScalingFactors)
            sf0_x.r, sf0_x.k, sf0_x.delta = r, k, X.radius
            sf0_x._log_lam = np.zeros(r + 1)
            sf0_y = ScalingFactors.__new__(ScalingFactors)
            sf0_y.r, sf0_y.k, sf0_y.delta = r, k, Y.radius
            sf0_y._log_lam = np.zeros(r + 1)

            U_n = lm.build_basis_lf(X, sf0_x, balanced=False)
            V_n = lm.build_basis_lf(Y, sf0_y, balanced=False)
            B_n = lm.build_B_lf(X, Y, sf0_x, sf0_y, balanced=False)
            with np.errstate(over='ignore', invalid='ignore'):
                phi_n = U_n @ (B_n @ (V_n.T @ q))
            B_fac_n = lm.balance_factor(B_n)
            err_n = kern.relative_error(phi_n, phi_exact)

            rows.append({'r': r,
                         'stable_B': B_fac_s, 'stable_err': err_s,
                         'naive_B': B_fac_n, 'naive_err': err_n})
        return rows

    def run_helmholtz_table62(self, L_tilde_values=None, r_values=None,
                              k: float = 10.0, scale: float = 0.01,
                              N: int = 50, seed: int = 42) -> List[dict]:
        """Reproduce HelmholtzKernel2D Table 6.2.

        k=10, scale=0.01. Expected: stable B ≈ 0.81 for L_tilde=2. [HK Table 6.2]
        """
        if r_values is None:
            r_values = [10, 15, 20, 25, 30, 35, 40, 45, 50]
        if L_tilde_values is None:
            L_tilde_values = [2]

        X, Y, q = self._make_helmholtz_clusters(N, scale, seed)
        kern = HelmholtzKernel(k)
        phi_exact = kern.matvec(X, Y, q)

        rows = []
        for L_tilde in L_tilde_values:
            for r in r_values:
                lm = LeafMatrices(r, k)
                sf_x = ScalingFactors(r, k, X.radius)
                sf_y = ScalingFactors(r, k, Y.radius)

                U_s = lm.build_basis_lf(X, sf_x, True)
                V_s = lm.build_basis_lf(Y, sf_y, True)
                B_s = lm.build_B_lf(X, Y, sf_x, sf_y, True)
                phi_s = U_s @ (B_s @ (V_s.T @ q))
                B_fac_s = lm.balance_factor(B_s)
                err_s = kern.relative_error(phi_s, phi_exact)

                sf0_x = ScalingFactors.__new__(ScalingFactors)
                sf0_x.r, sf0_x.k, sf0_x.delta = r, k, X.radius
                sf0_x._log_lam = np.zeros(r + 1)
                sf0_y = ScalingFactors.__new__(ScalingFactors)
                sf0_y.r, sf0_y.k, sf0_y.delta = r, k, Y.radius
                sf0_y._log_lam = np.zeros(r + 1)

                U_n = lm.build_basis_lf(X, sf0_x, False)
                V_n = lm.build_basis_lf(Y, sf0_y, False)
                B_n = lm.build_B_lf(X, Y, sf0_x, sf0_y, False)
                with np.errstate(over='ignore', invalid='ignore'):
                    phi_n = U_n @ (B_n @ (V_n.T @ q))
                B_fac_n = lm.balance_factor(B_n)
                err_n = kern.relative_error(phi_n, phi_exact)

                rows.append({'L_tilde': L_tilde, 'r': r,
                             'stable_B': B_fac_s, 'stable_err': err_s,
                             'naive_B': B_fac_n, 'naive_err': err_n})
        return rows

    def run_helmholtz_table63(self, L_tilde_values=None, r_values=None,
                              k: float = 100.0, scale: float = 0.01,
                              N: int = 50, seed: int = 42) -> List[dict]:
        """Reproduce HelmholtzKernel2D Table 6.3.

        k=100, scale=0.01. Expected: stable B ≈ 0.27. [HK Table 6.3]
        """
        if r_values is None:
            r_values = [100, 120, 140, 160, 180]
        if L_tilde_values is None:
            L_tilde_values = [2]

        return self.run_helmholtz_table62(L_tilde_values, r_values, k, scale, N, seed)

    @staticmethod
    def _cauchy_UBV(X: PointSet, Y: PointSet, r: int,
                    balanced: bool = True) -> Tuple[np.ndarray, ...]:
        """Low-rank Cauchy-kernel factorization K ≈ U @ B @ V^T. [M2D §2]"""
        from scipy.special import gammaln as _gl
        d = X.center - Y.center
        if abs(d) < 1e-15:
            return (np.zeros((len(X), r + 1), dtype=complex),
                    np.zeros((r + 1, r + 1), dtype=complex),
                    np.zeros((len(Y), r + 1), dtype=complex))

        u = X.points - X.center
        v = Y.points - Y.center
        delta_x = max(X.radius, 1e-300)
        delta_y = max(Y.radius, 1e-300)
        log_dx = np.log(delta_x)
        log_dy = np.log(delta_y)
        log_d  = np.log(abs(d))
        arg_d  = float(np.angle(d))

        M, N_pts = len(X), len(Y)
        U = np.zeros((M, r + 1), dtype=complex)
        V = np.zeros((N_pts, r + 1), dtype=complex)
        for p in range(r + 1):
            with np.errstate(over='ignore', invalid='ignore'):
                U[:, p] = (u / delta_x) ** p if balanced else u ** p
        for q in range(r + 1):
            with np.errstate(over='ignore', invalid='ignore'):
                V[:, q] = (v / delta_y) ** q if balanced else v ** q

        max_n = 2 * r
        log_fac = _gl(np.arange(max_n + 2) + 1)

        B = np.zeros((r + 1, r + 1), dtype=complex)
        for p in range(r + 1):
            sign_p = (-1) ** p
            for q in range(r + 1):
                pq1 = p + q + 1
                log_C = log_fac[p + q] - log_fac[p] - log_fac[q]
                if balanced:
                    log_abs = log_C + p * log_dx + q * log_dy - pq1 * log_d
                    phase = sign_p * np.exp(-1j * pq1 * arg_d)
                    if log_abs > 709:
                        B[p, q] = np.inf * phase
                    elif log_abs < -709:
                        B[p, q] = 0.0
                    else:
                        B[p, q] = np.exp(log_abs) * phase
                else:
                    log_abs_n = log_C - pq1 * log_d
                    phase = sign_p * np.exp(-1j * pq1 * arg_d)
                    if log_abs_n > 709:
                        B[p, q] = np.inf * phase
                    elif log_abs_n < -709:
                        B[p, q] = 0.0
                    else:
                        B[p, q] = np.exp(log_abs_n) * phase

        return U, B, V

    @staticmethod
    def _log_UBV(X: PointSet, Y: PointSet, r: int,
                 balanced: bool = True) -> Tuple[np.ndarray, ...]:
        """Low-rank log-kernel factorization K ≈ U @ B @ V^T. [M2D §2]"""
        from scipy.special import gammaln as _gl
        d = X.center - Y.center
        if abs(d) < 1e-15:
            return (np.zeros((len(X), r + 1), dtype=complex),
                    np.zeros((r + 1, r + 1), dtype=complex),
                    np.zeros((len(Y), r + 1), dtype=complex))

        u = X.points - X.center
        v = Y.points - Y.center
        delta_x = max(X.radius, 1e-300)
        delta_y = max(Y.radius, 1e-300)
        log_dx = np.log(delta_x)
        log_dy = np.log(delta_y)
        log_d  = np.log(abs(d))
        arg_d  = float(np.angle(d))

        M, N_pts = len(X), len(Y)
        U = np.zeros((M, r + 1), dtype=complex)
        V = np.zeros((N_pts, r + 1), dtype=complex)
        for p in range(r + 1):
            with np.errstate(over='ignore', invalid='ignore'):
                U[:, p] = (u / delta_x) ** p if balanced else u ** p
        for q in range(r + 1):
            with np.errstate(over='ignore', invalid='ignore'):
                V[:, q] = (v / delta_y) ** q if balanced else v ** q

        max_n = 2 * r
        log_fac = _gl(np.arange(max_n + 2) + 1)

        B = np.zeros((r + 1, r + 1), dtype=complex)
        B[0, 0] = np.log(d)
        for p in range(r + 1):
            for q in range(r + 1):
                pq = p + q
                if pq == 0:
                    continue
                sign_pp1 = (-1) ** (p + 1)
                log_C = log_fac[pq] - log_fac[p] - log_fac[q]
                if balanced:
                    log_abs = log_C + p * log_dx + q * log_dy - pq * log_d - np.log(pq)
                    phase = sign_pp1 * np.exp(-1j * pq * arg_d)
                    if log_abs > 709:
                        B[p, q] = np.inf * phase
                    elif log_abs < -709:
                        B[p, q] = 0.0
                    else:
                        B[p, q] = np.exp(log_abs) * phase
                else:
                    log_abs_n = log_C - pq * log_d - np.log(pq)
                    phase = sign_pp1 * np.exp(-1j * pq * arg_d)
                    if log_abs_n > 709:
                        B[p, q] = np.inf * phase
                    elif log_abs_n < -709:
                        B[p, q] = 0.0
                    else:
                        B[p, q] = np.exp(log_abs_n) * phase

        return U, B, V

    def run_multipole2d_table61(self, r_values=None, N: int = 80,
                                scale: float = 1e-4, seed: int = 42) -> List[dict]:
        """Reproduce Multipole2D Table 6.1: Cauchy kernel, scale=1e-4. [M2D Table 6.1]"""
        if r_values is None:
            r_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        rng = np.random.default_rng(seed)
        r_x = np.sqrt(rng.uniform(0, 1, N))
        th_x = rng.uniform(0, 2 * np.pi, N)
        r_y = np.sqrt(rng.uniform(0, 1, N))
        th_y = rng.uniform(0, 2 * np.pi, N)
        X = PointSet(scale * r_x * np.exp(1j * th_x))
        Y = PointSet(scale * (6.0 + r_y * np.exp(1j * th_y)))

        q = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        K_exact = HelmholtzKernel.cauchy_matrix(X, Y)
        phi_exact = K_exact @ q

        rows = []
        for r in r_values:
            err_b = err_n = np.inf
            try:
                U_b, B_b, V_b = self._cauchy_UBV(X, Y, r, balanced=True)
                with np.errstate(over='ignore', invalid='ignore'):
                    phi_b = U_b @ (B_b @ (V_b.T @ q))
                if np.all(np.isfinite(phi_b)):
                    err_b = float(np.linalg.norm(phi_b - phi_exact) /
                                  np.linalg.norm(phi_exact))
            except Exception:
                pass
            try:
                U_n, B_n, V_n = self._cauchy_UBV(X, Y, r, balanced=False)
                with np.errstate(over='ignore', invalid='ignore'):
                    phi_n = U_n @ (B_n @ (V_n.T @ q))
                if np.all(np.isfinite(phi_n)):
                    err_n = float(np.linalg.norm(phi_n - phi_exact) /
                                  np.linalg.norm(phi_exact))
            except Exception:
                pass

            rows.append({'r': r, 'balanced_err': err_b, 'naive_err': err_n})
        return rows

    def run_multipole2d_table62(self, r_values=None, N: int = 80,
                                scale: float = 100.0, seed: int = 42) -> List[dict]:
        """Reproduce Multipole2D Table 6.2: log kernel, scale=100. [M2D Table 6.2]"""
        if r_values is None:
            r_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]

        rng = np.random.default_rng(seed)
        r_x = np.sqrt(rng.uniform(0, 1, N))
        th_x = rng.uniform(0, 2 * np.pi, N)
        r_y = np.sqrt(rng.uniform(0, 1, N))
        th_y = rng.uniform(0, 2 * np.pi, N)
        X = PointSet(scale * (6.0 + r_x * np.exp(1j * th_x)))
        Y = PointSet(scale * r_y * np.exp(1j * th_y))

        q = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        K_exact = HelmholtzKernel.log_matrix(X, Y)
        phi_exact = K_exact @ q

        rows = []
        for r in r_values:
            err_b = err_n = np.inf
            try:
                U_b, B_b, V_b = self._log_UBV(X, Y, r, balanced=True)
                with np.errstate(over='ignore', invalid='ignore'):
                    phi_b = U_b @ (B_b @ (V_b.T @ q))
                if np.all(np.isfinite(phi_b)):
                    err_b = float(np.linalg.norm(phi_b - phi_exact) /
                                  np.linalg.norm(phi_exact))
            except Exception:
                pass
            try:
                U_n, B_n, V_n = self._log_UBV(X, Y, r, balanced=False)
                with np.errstate(over='ignore', invalid='ignore'):
                    phi_n = U_n @ (B_n @ (V_n.T @ q))
                if np.all(np.isfinite(phi_n)):
                    err_n = float(np.linalg.norm(phi_n - phi_exact) /
                                  np.linalg.norm(phi_exact))
            except Exception:
                pass

            rows.append({'r': r, 'balanced_err': err_b, 'naive_err': err_n})
        return rows

    @staticmethod
    def plot_error_vs_rank(rows: List[dict],
                           key_b: str = 'stable_err',
                           key_n: str = 'naive_err',
                           title: str = 'Error vs rank r') -> object:
        """Log-scale plot of balanced vs naive error. Returns matplotlib Figure."""
        import matplotlib.pyplot as plt

        r_vals = [row['r'] for row in rows]
        err_b = [row.get(key_b, np.nan) for row in rows]
        err_n = [row.get(key_n, np.nan) for row in rows]

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.semilogy(r_vals, err_b, 'b-o', lw=2, ms=6, label='Balanced (estable)')

        finite_mask = np.isfinite(err_n)
        if np.any(finite_mask):
            r_fin = [r for r, m in zip(r_vals, finite_mask) if m]
            e_fin = [e for e, m in zip(err_n, finite_mask) if m]
            ax.semilogy(r_fin, e_fin, 'r-s', lw=2, ms=6, label='Naive (regular)')
        inf_mask = ~np.array(finite_mask)
        if np.any(inf_mask):
            r_inf = [r for r, m in zip(r_vals, inf_mask) if m]
            y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
            ax.plot(r_inf, [y_top] * len(r_inf), 'rv', ms=12,
                    label='Naive = Inf/NaN', zorder=5)

        ax.set_xlabel('Orden de expansion $r$', fontsize=12)
        ax.set_ylabel('Error relativo $\\|\\phi_{aprox} - \\phi\\| / \\|\\phi\\|$', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_balance_factor(rows: List[dict],
                            key_b: str = 'stable_B',
                            key_n: str = 'naive_B',
                            title: str = 'Factor de balance B vs rango r') -> object:
        """Log-scale plot of stable vs naive B-factor. Returns matplotlib Figure."""
        import matplotlib.pyplot as plt

        r_vals = [row['r'] for row in rows]
        B_b = [row.get(key_b, np.nan) for row in rows]
        B_n = [row.get(key_n, np.nan) for row in rows]

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.semilogy(r_vals, B_b, 'b-o', lw=2, ms=6, label='$\\mathcal{B}$ estable')
        finite_mask = np.isfinite(B_n)
        if np.any(finite_mask):
            r_f = [r for r, m in zip(r_vals, finite_mask) if m]
            B_f = [B for B, m in zip(B_n, finite_mask) if m]
            ax.semilogy(r_f, B_f, 'r-s', lw=2, ms=6, label='$\\mathcal{B}$ regular')
        if np.any(~np.array(finite_mask)):
            r_inf = [r for r, m in zip(r_vals, finite_mask) if not m]
            y_top = max([b for b in B_n if np.isfinite(b)] or [1.0]) * 1e3
            ax.plot(r_inf, [y_top] * len(r_inf), 'rv', ms=12,
                    label='$\\mathcal{B}$ regular = Inf')

        ax.set_xlabel('Orden de expansion $r$', fontsize=12)
        ax.set_ylabel('$\\mathcal{B} = \\|B\\|_{\\max}$', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# StabilityAnalyzer
# ---------------------------------------------------------------------------

class StabilityAnalyzer:
    """Diagnostic tools for verifying the stability theory from [HK §2].

    Key verifiable claims:
    - ||U||_max <= 1 when balanced=True  [HK Theorem 2.7]
    - ||B||_max = O(1) when balanced, grows exponentially when naive
    - Backward error grows logarithmically with N (not linearly)
    """

    def __init__(self, k: float, r: int):
        self.k = float(k)
        self.r = int(r)
        self._lm = LeafMatrices(r, k)

    def verify_norm_bounds(self, X: PointSet, Y: PointSet) -> dict:
        """Check ||U||_max <= 1 for balanced basis [HK Theorem 2.7]."""
        sf_x = ScalingFactors(self.r, self.k, X.radius)
        sf_y = ScalingFactors(self.r, self.k, Y.radius)

        U_bal = self._lm.build_basis_lf(X, sf_x, balanced=True)
        V_bal = self._lm.build_basis_lf(Y, sf_y, balanced=True)
        B_bal = self._lm.build_B_lf(X, Y, sf_x, sf_y, balanced=True)

        U_nai = self._lm.build_basis_lf(X, sf_x, balanced=False)
        B_nai = self._lm.build_B_lf(X, Y, sf_x, sf_y, balanced=False)

        return {
            'U_max_balanced': float(np.max(np.abs(U_bal))),
            'V_max_balanced': float(np.max(np.abs(V_bal))),
            'B_max_balanced': LeafMatrices.balance_factor(B_bal),
            'U_max_naive':    float(np.max(np.abs(U_nai[np.isfinite(U_nai)]))),
            'B_max_naive':    LeafMatrices.balance_factor(B_nai),
            'theorem_satisfied': float(np.max(np.abs(U_bal))) <= 1.0 + 1e-10,
        }

    def sweep_rank(self, X: PointSet, Y: PointSet,
                   q: np.ndarray, r_values: List[int]) -> List[dict]:
        """Compare stable vs naive accuracy across a range of expansion orders r."""
        kern = HelmholtzKernel(self.k)
        phi_exact = kern.matvec(X, Y, q)
        rows = []
        bs = BenchmarkSuite()
        for r in r_values:
            lm = LeafMatrices(r, self.k)
            sf_x = ScalingFactors(r, self.k, X.radius)
            sf_y = ScalingFactors(r, self.k, Y.radius)
            U_b = lm.build_basis_lf(X, sf_x, True)
            V_b = lm.build_basis_lf(Y, sf_y, True)
            B_b = lm.build_B_lf(X, Y, sf_x, sf_y, True)
            phi_b = U_b @ (B_b @ (V_b.T @ q))
            err_b = kern.relative_error(phi_b, phi_exact)

            U_n = lm.build_basis_lf(X, sf_x, False)
            V_n = lm.build_basis_lf(Y, sf_y, False)
            B_n = lm.build_B_lf(X, Y, sf_x, sf_y, False)
            with np.errstate(over='ignore', invalid='ignore'):
                phi_n = U_n @ (B_n @ (V_n.T @ q))
            err_n = kern.relative_error(phi_n, phi_exact)

            rows.append({'r': r,
                         'stable_B': lm.balance_factor(B_b), 'stable_err': err_b,
                         'naive_B': lm.balance_factor(B_n), 'naive_err': err_n})
        return rows

    def find_overflow_threshold(self, X: PointSet, Y: PointSet,
                                r_max: int = 200, step: int = 10) -> int:
        """Return first r at which naive B overflows to Inf."""
        for r in range(10, r_max + 1, step):
            lm = LeafMatrices(r, self.k)
            sf0 = ScalingFactors.__new__(ScalingFactors)
            sf0.r, sf0.k, sf0.delta = r, self.k, X.radius
            sf0._log_lam = np.zeros(r + 1)
            B_n = lm.build_B_lf(X, Y, sf0, sf0, balanced=False)
            if not np.all(np.isfinite(B_n)):
                return r
        return r_max + 1
