"""
core.py — PointSet, ScalingFactors, HelmholtzKernel

References:
  [HK]  HelmholtzKernel2D.pdf — Michelle, Ou, Xia; preprint 2024
  [M2D] Multipole2D.pdf — Ou, Michelle, Xia; SIAM J. Matrix Anal. Appl. 46(1), 2025
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import gammaln, hankel1, jv


# ---------------------------------------------------------------------------
# 1. PointSet
# ---------------------------------------------------------------------------

class PointSet:
    """A set of points in the complex plane with its bounding disk.

    Attributes
    ----------
    points : ndarray of complex, shape (N,)
    center : complex  — centroid o_x [HK §1]
    radius : float    — bounding radius delta_x = max|points - center|
    """

    def __init__(self, points: np.ndarray, center: Optional[complex] = None):
        self.points = np.asarray(points, dtype=complex).ravel()
        self.center = complex(center) if center is not None else complex(np.mean(self.points))
        diffs = np.abs(self.points - self.center)
        self.radius = float(np.max(diffs)) if len(diffs) > 0 else 0.0

    @classmethod
    def random_uniform(cls, n: int, center: complex = 0+0j,
                       radius: float = 1.0, rng=None) -> PointSet:
        """N uniform random points in a disk of given center and radius."""
        rng = np.random.default_rng(rng)
        r = radius * np.sqrt(rng.uniform(0, 1, n))
        theta = rng.uniform(0, 2 * np.pi, n)
        pts = center + r * np.exp(1j * theta)
        return cls(pts, center)

    @classmethod
    def random_cluster(cls, n: int, center: complex = 0+0j,
                       spread: float = 0.1, rng=None) -> PointSet:
        """N Gaussian-distributed points around center."""
        rng = np.random.default_rng(rng)
        pts = center + spread * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        return cls(pts)

    def scaled(self, scale: float) -> PointSet:
        """Return a new PointSet scaled by `scale` (points and center)."""
        return PointSet(scale * self.points, scale * self.center)

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        return (f"PointSet(N={len(self)}, center={self.center:.4g}, "
                f"radius={self.radius:.4g})")


# ---------------------------------------------------------------------------
# 2. ScalingFactors
# ---------------------------------------------------------------------------

class ScalingFactors:
    """Precomputed lambda_{x,p} = max{1, |p|! * (2/(k*delta))^|p|} for p=0,...,r.

    Computed on log scale via gammaln to handle large r without overflow.
    [HK Eq. 2.17] — the key to achieving ||U||_max <= 1.

    Constructor order: ScalingFactors(r, k, delta) — r first.
    """

    def __init__(self, r: int, k: float, delta: float):
        self.r = r
        self.k = k
        self.delta = delta
        self._log_lam = self._compute_log()

    def _compute_log(self) -> np.ndarray:
        r, k, delta = self.r, self.k, self.delta
        kd = k * delta
        log_lam = np.zeros(r + 1, dtype=float)
        for p in range(1, r + 1):
            log_val = gammaln(p + 1) + p * np.log(2.0 / max(kd, 1e-300))
            log_lam[p] = max(0.0, log_val)
        return log_lam

    def __getitem__(self, p: int) -> float:
        """Return lambda_{|p|} as a float."""
        return float(np.exp(min(self._log_lam[abs(p)], 709.0)))

    def log_lam(self, p: int) -> float:
        """Return log(lambda_{|p|}), always finite."""
        return self._log_lam[abs(p)]

    def as_array(self) -> np.ndarray:
        """Return lambda values for p=0,...,r."""
        return np.exp(np.minimum(self._log_lam, 709.0))

    def log_array(self) -> np.ndarray:
        """Return log(lambda_p) for p=0,...,r, always finite."""
        return self._log_lam.copy()

    def ratio(self, p_new: int, p_old: int) -> float:
        """lambda_{p_new} / lambda_{p_old} in linear scale."""
        log_ratio = self._log_lam[abs(p_new)] - self._log_lam[abs(p_old)]
        return float(np.exp(np.clip(log_ratio, -709, 709)))


# ---------------------------------------------------------------------------
# 3. HelmholtzKernel — direct O(MN) evaluator + kernel factories
# ---------------------------------------------------------------------------

class HelmholtzKernel:
    """Direct O(MN) evaluation of K[i,j] = H_0^(1)(k|x_i - y_j|).

    Used as ground truth for error checking. Also provides static factories
    for the Cauchy and logarithmic kernels (Multipole2D benchmarks).
    """

    def __init__(self, k: float):
        self.k = float(k)

    def matrix(self, X: PointSet, Y: PointSet) -> np.ndarray:
        """Return (M x N) kernel matrix K[i,j] = H_0^(1)(k|x_i - y_j|)."""
        xi = X.points[:, None]
        yj = Y.points[None, :]
        dist = np.abs(xi - yj)
        dist = np.maximum(dist, 1e-15)
        return hankel1(0, self.k * dist)

    def matvec(self, X: PointSet, Y: PointSet, q: np.ndarray) -> np.ndarray:
        """Return phi = K @ q (direct sum, O(MN))."""
        return self.matrix(X, Y) @ np.asarray(q, dtype=complex)

    def relative_error(self, phi_approx: np.ndarray, phi_exact: np.ndarray) -> float:
        """Return ||phi_approx - phi_exact||_2 / ||phi_exact||_2."""
        num = np.linalg.norm(phi_approx - phi_exact)
        den = np.linalg.norm(phi_exact)
        return float(num / den) if den > 0 else float(num)

    @staticmethod
    def cauchy_matrix(X: PointSet, Y: PointSet) -> np.ndarray:
        """K[i,j] = 1 / (x_i - y_j) — generalized Cauchy kernel [M2D §2]."""
        xi = X.points[:, None]
        yj = Y.points[None, :]
        diff = xi - yj
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(np.abs(diff) > 1e-15, 1.0 / diff, 0.0)

    @staticmethod
    def log_matrix(X: PointSet, Y: PointSet) -> np.ndarray:
        """K[i,j] = log(x_i - y_j) — logarithmic kernel [M2D §2]."""
        xi = X.points[:, None]
        yj = Y.points[None, :]
        diff = xi - yj
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(np.abs(diff) > 1e-15, np.log(diff), 0.0)
