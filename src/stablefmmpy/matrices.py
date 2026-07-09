"""
matrices.py — LeafMatrices: builds U, V, B, R matrices for one pair of boxes.

References:
  [HK]  HelmholtzKernel2D.pdf — Michelle, Ou, Xia; preprint 2024
  [M2D] Multipole2D.pdf — Ou, Michelle, Xia; SIAM J. Matrix Anal. Appl. 46(1), 2025
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import gammaln, hankel1, jv

from .core import PointSet, ScalingFactors


class LeafMatrices:
    """Builds leaf-level factorization matrices for a single pair of boxes.

    Implements both LF (low-frequency) and HF (high-frequency) regimes.
    The `balanced=True` flag applies the lambda scaling [HK Eq. 2.17],
    giving ||U||_max <= 1; `balanced=False` omits it, causing overflow.

    Notation (consistent with [HK §2]):
      r      : expansion order (rank parameter)
      k      : wavenumber
      V^(L)  : source-side basis matrix  (N x (2r+1))
      U^(L)  : target-side basis matrix  (M x (2r+1))
      B^(l)  : M2L conversion matrix     ((2r+1) x (2r+1))
      R^(l)  : inter-level restriction   ((2r+1) x (2r+1))

    Factorization: K ≈ U @ B @ V^T, so phi = U @ B @ V^T @ q

    Constructor order: LeafMatrices(r, k) — r first.
    """

    def __init__(self, r: int, k: float):
        self.r = r
        self.k = k

    def regime(self, delta: float) -> str:
        """Return 'lf' if k*delta <= r/e (low-frequency), else 'hf'."""
        return 'lf' if self.k * delta <= self.r / np.e else 'hf'

    # ---- Low-frequency regime -----------------------------------------------

    def build_basis_lf(self, ps: PointSet, sf: ScalingFactors,
                       balanced: bool = True) -> np.ndarray:
        """Build basis matrix (M x (2r+1)) via Miller backward recurrence.

        Entry [i, p+r] = J_{|p|}(k*|z_i|) * e^{ip*arg(z_i)} / lambda_{|p|}
        where z_i = k*(point_i - center).  [HK Eq. 2.9, 4.5]
        """
        M = len(ps.points)
        n_cols = 2 * self.r + 1
        basis = np.zeros((M, n_cols), dtype=complex)
        log_lam = sf.log_array() if balanced else np.zeros(self.r + 1)

        for i, pt in enumerate(ps.points):
            z_c = self.k * (pt - ps.center)
            z_abs = float(abs(z_c))
            z_arg = float(np.angle(z_c))

            if z_abs > 1e-30:
                log_psi = np.full(self.r + 2, -np.inf)
                log_psi[self.r] = 0.0

                for p in range(self.r, 0, -1):
                    log_A = np.log(2.0 * p / z_abs) + log_psi[p]
                    log_B_p = log_psi[p + 1]
                    if log_A >= log_B_p:
                        corr = (np.log1p(-np.exp(log_B_p - log_A))
                                if log_B_p > -np.inf else 0.0)
                        log_psi[p - 1] = log_A + corr
                    else:
                        corr = (np.log1p(-np.exp(log_A - log_B_p))
                                if log_A > -np.inf else 0.0)
                        log_psi[p - 1] = log_B_p + corr

                J0_exact = float(jv(0, z_abs).real)
                log_J0 = (np.log(abs(J0_exact))
                          if abs(J0_exact) > 1e-300 else -np.inf)
                log_norm = log_J0 - log_psi[0]
                log_J = log_psi[:self.r + 1] + log_norm
            else:
                log_J = np.full(self.r + 1, -np.inf)
                log_J[0] = 0.0

            for idx, p in enumerate(range(-self.r, self.r + 1)):
                ap = abs(p)
                sign = (-1) ** p if p < 0 else 1.0
                phase = np.exp(1j * p * z_arg)
                log_entry = log_J[ap] + log_lam[ap]
                if log_entry < -709:
                    basis[i, idx] = 0.0 + 0.0j
                else:
                    basis[i, idx] = sign * np.exp(log_entry) * phase

        return basis

    def build_B_lf(self, X: PointSet, Y: PointSet,
                   sf_x: ScalingFactors, sf_y: ScalingFactors,
                   balanced: bool = True) -> np.ndarray:
        """Build M2L matrix B ((2r+1) x (2r+1)) via upward Hankel recurrence.

        B[p, q] = (-1)^p * H_{p+q}^(1)(k*d) / (lambda_p * lambda_q)  [balanced]
        where d = |o_X - o_Y|.  [HK Eq. 2.10; HK Lemma 2.3]
        """
        d_c = X.center - Y.center
        dist = abs(d_c)
        z = self.k * dist
        z_s = max(z, 1e-15)
        theta_d = float(np.angle(d_c))

        H = np.zeros(2 * self.r + 2, dtype=complex)
        H[0] = hankel1(0, z_s)
        H[1] = hankel1(1, z_s)
        with np.errstate(over='ignore', invalid='ignore'):
            for n in range(1, 2 * self.r):
                H[n + 1] = (2.0 * n / z_s) * H[n] - H[n - 1]

        log_lam_x = sf_x.log_array() if balanced else np.zeros(self.r + 1)
        log_lam_y = sf_y.log_array() if balanced else np.zeros(self.r + 1)

        n2r = 2 * self.r + 1
        B = np.zeros((n2r, n2r), dtype=complex)

        for ip, p in enumerate(range(-self.r, self.r + 1)):
            for iq, q in enumerate(range(-self.r, self.r + 1)):
                pq = p + q
                apq = abs(pq)
                if apq > 2 * self.r:
                    continue

                H_pq = H[apq] * ((-1) ** apq if pq < 0 else 1)
                sign_p = (-1) ** p
                phase_B = np.exp(-1j * pq * theta_d)

                if not balanced:
                    B[ip, iq] = sign_p * phase_B * H_pq
                    continue

                ll_sum = log_lam_x[abs(p)] + log_lam_y[abs(q)]
                use_direct = False
                if ll_sum <= 709.0:
                    with np.errstate(over='ignore', invalid='ignore'):
                        lam_prod = np.exp(ll_sum)
                        val = H_pq / lam_prod
                    use_direct = np.isfinite(val)

                if use_direct:
                    B[ip, iq] = sign_p * phase_B * val
                else:
                    if apq == 0:
                        log_absH = np.log(max(abs(H[0]), 1e-300))
                        ph_H = H[0] / (abs(H[0]) + 1e-300)
                    else:
                        log_absH = (gammaln(apq) - np.log(np.pi)
                                    + apq * np.log(2.0 / z_s))
                        ph_H = -1j * ((-1) ** apq if pq < 0 else 1)
                    log_ratio = log_absH - ll_sum
                    if log_ratio < -709:
                        B[ip, iq] = 0.0 + 0.0j
                    elif log_ratio > 709:
                        B[ip, iq] = sign_p * phase_B * np.inf * ph_H
                    else:
                        B[ip, iq] = sign_p * phase_B * np.exp(log_ratio) * ph_H

        return B

    def build_R_lf(self, child_ps: PointSet, parent_ps: PointSet,
                   sf_child: ScalingFactors, sf_parent: ScalingFactors,
                   balanced: bool = True) -> np.ndarray:
        """Build inter-level restriction matrix R ((2r+1) x (2r+1)).

        R[p, l] = (lambda_parent_p / lambda_child_l) * J_{p-l}(k*d) * e^{i(p-l)*theta}
        [HK §3 Theorem 3.1; M2D §3]
        """
        d_c = child_ps.center - parent_ps.center
        d_abs = float(abs(d_c))
        theta = float(np.angle(self.k * d_c)) if d_abs > 1e-15 else 0.0
        kd = self.k * d_abs

        n2r = 2 * self.r + 1
        R = np.zeros((n2r, n2r), dtype=complex)

        log_lam_p = sf_parent.log_array() if balanced else np.zeros(self.r + 1)
        log_lam_c = sf_child.log_array()  if balanced else np.zeros(self.r + 1)

        for ip, p in enumerate(range(-self.r, self.r + 1)):
            for il, l in enumerate(range(-self.r, self.r + 1)):
                m = p - l
                J_m = float(jv(abs(m), kd).real) * ((-1) ** abs(m) if m < 0 else 1)
                phase = np.exp(1j * m * theta) if kd > 0 else 1.0
                log_ratio = log_lam_p[abs(p)] - log_lam_c[abs(l)]
                with np.errstate(over='ignore'):
                    ratio = np.exp(min(log_ratio, 709.0))
                R[ip, il] = ratio * J_m * phase

        return R

    # ---- High-frequency regime (DFT diagonal basis) -------------------------

    def build_basis_hf(self, ps: PointSet, sign: int = +1) -> np.ndarray:
        """Build HF basis matrix (M x (2r+1)) using equispaced plane waves.

        [HK §2.3; Cecka & Darve 2013]
        sign = +1 for V (source side), -1 for U (target side).
        """
        M = len(ps.points)
        n2r = 2 * self.r + 1
        basis = np.zeros((M, n2r), dtype=complex)
        thetas = 2.0 * np.pi * np.arange(n2r) / n2r

        for i, pt in enumerate(ps.points):
            z_c = self.k * (pt - ps.center)
            z_abs = float(abs(z_c))
            z_arg = float(np.angle(z_c))
            basis[i, :] = np.exp(sign * 1j * z_abs * np.cos(thetas - z_arg))

        return basis

    def build_B_hf(self, X: PointSet, Y: PointSet) -> np.ndarray:
        """Build HF M2L matrix B ((2r+1) x (2r+1)): diagonal, always finite.

        [HK §2.3]
        """
        n2r = 2 * self.r + 1
        d_c = X.center - Y.center
        d_abs = float(abs(d_c))
        theta = float(np.angle(d_c))
        thetas = 2.0 * np.pi * np.arange(n2r) / n2r

        B = np.zeros((n2r, n2r), dtype=complex)
        for p in range(n2r):
            arg = self.k * d_abs * np.cos(thetas[p] - theta)
            arg = max(abs(arg), 1e-15)
            B[p, p] = hankel1(0, arg)

        return B

    def build_R_hf(self, child_ps: PointSet, parent_ps: PointSet) -> np.ndarray:
        """Build HF inter-level restriction R ((2r+1) x (2r+1)): diagonal phase shift.

        [HK §3.2]
        """
        n2r = 2 * self.r + 1
        d_c = child_ps.center - parent_ps.center
        thetas = 2.0 * np.pi * np.arange(n2r) / n2r

        R = np.zeros((n2r, n2r), dtype=complex)
        for p in range(n2r):
            R[p, p] = np.exp(1j * self.k * abs(d_c) * np.cos(thetas[p] - np.angle(d_c)))

        return R

    # ---- High-level factorize API -------------------------------------------

    def factorize(self, X: PointSet, Y: PointSet, balanced: bool = True,
                  regime: str = 'auto') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (U, B, V) such that K ≈ U @ B @ V.T."""
        if regime == 'auto':
            regime = self.regime(max(X.radius, Y.radius))

        if regime == 'lf':
            sf_x = ScalingFactors(self.r, self.k, X.radius)
            sf_y = ScalingFactors(self.r, self.k, Y.radius)
            U = self.build_basis_lf(X, sf_x, balanced)
            V = self.build_basis_lf(Y, sf_y, balanced)
            B = self.build_B_lf(X, Y, sf_x, sf_y, balanced)
        else:
            U = self.build_basis_hf(X, sign=-1)
            V = self.build_basis_hf(Y, sign=+1)
            B = self.build_B_hf(X, Y)

        return U, B, V

    @staticmethod
    def balance_factor(B: np.ndarray) -> float:
        """Return ||B||_max = max|B[p,q]| — the B metric in the paper tables."""
        with np.errstate(invalid='ignore'):
            finite_vals = np.abs(B[np.isfinite(B)])
        if len(finite_vals) == 0:
            return np.inf
        has_inf = not np.all(np.isfinite(B))
        return np.inf if has_inf else float(np.max(np.abs(B)))
