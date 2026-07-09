"""
solver.py — FMMSolver: full FMM algorithm for phi = K q.

References:
  [HK]  HelmholtzKernel2D.pdf — Michelle, Ou, Xia; preprint 2024
  [M2D] Multipole2D.pdf — Ou, Michelle, Xia; SIAM J. Matrix Anal. Appl. 46(1), 2025
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .core import PointSet, HelmholtzKernel, ScalingFactors
from .matrices import LeafMatrices
from .tree import QuadTree, FMMNode


class FMMSolver:
    """Implements the FMM matrix-vector product phi = K q in O((M+N)*r) time.

    Two internal modes:
    - _solve_single():    Multipole2D Algorithm 4.1 (single-regime FMM)
    - _solve_wideband():  HelmholtzKernel2D Algorithm 4.1 (LF + HF combined)

    solve() always calls _solve_wideband(), which selects the regime per leaf.

    References:
      [M2D §4 Algorithm 4.1]  — single regime
      [HK §4 Algorithm 4.1]   — wideband (two regimes)
    """

    def __init__(self, k: float, r: int, tau: float = 0.6, N0: int = 32,
                 balanced: bool = True):
        self.k = float(k)
        self.r = int(r)
        self.tau = tau
        self.N0 = N0
        self.balanced = balanced
        self._lm = LeafMatrices(r, k)

    def solve(self, X: PointSet, Y: PointSet, q: np.ndarray) -> np.ndarray:
        """Compute phi = K q approximately via adaptive FMM. Builds tree internally."""
        tree = QuadTree()
        tree.build(Y.points, X.points, tau=self.tau, N0=self.N0)
        return self._solve_wideband(tree, X, Y, q)

    def _solve_single(self, tree: QuadTree, X: PointSet, Y: PointSet,
                      q: np.ndarray) -> np.ndarray:
        """Leaf-only single-regime FMM [M2D §4 Algorithm 4.1].

        Flat (2-level) structure — no M2M/L2L passes between tree levels.
        For each target leaf A:
          phi[A] += U_A @ sum_{B in interaction_list(A)} B_AB @ v_B   [M2L far-field]
          phi[A] += sum_{C in near_list(A) + {A}} K_AC @ q[C.src_idx] [P2P near-field]
        where v_B = V_B^T @ q[B.src_idx] is precomputed in Phase 1.
        """
        src_pts = Y.points
        tgt_pts = X.points
        phi = np.zeros(len(tgt_pts), dtype=complex)
        kern = HelmholtzKernel(self.k)
        n2r1 = 2 * self.r + 1

        leaf_cache: Dict[int, dict] = {}
        for node in tree.postorder():
            if not node.is_leaf():
                continue
            s_idx = node.src_idx
            if len(s_idx) == 0:
                leaf_cache[node.box_id] = {'v_vec': None, 'src_ps': None, 'sf': None}
                continue
            src_sub = PointSet(src_pts[s_idx])
            delta_s = max(src_sub.radius, 1e-15)
            sf_s = ScalingFactors(self.r, self.k, delta_s)
            V = self._lm.build_basis_lf(src_sub, sf_s, self.balanced)
            leaf_cache[node.box_id] = {
                'v_vec': V.T @ q[s_idx],
                'src_ps': src_sub,
                'sf': sf_s,
            }

        for node in tree.postorder():
            if not node.is_leaf():
                continue
            t_idx = node.tgt_idx
            if len(t_idx) == 0:
                continue

            tgt_sub = PointSet(tgt_pts[t_idx])
            delta_t = max(tgt_sub.radius, 1e-15)
            sf_t = ScalingFactors(self.r, self.k, delta_t)
            U = self._lm.build_basis_lf(tgt_sub, sf_t, self.balanced)

            u_acc = np.zeros(n2r1, dtype=complex)
            for partner in node.interaction_list:
                pdata = leaf_cache.get(partner.box_id, {})
                if pdata.get('v_vec') is None:
                    continue
                B = self._lm.build_B_lf(
                    tgt_sub, pdata['src_ps'], sf_t, pdata['sf'], self.balanced)
                u_acc += B @ pdata['v_vec']
            phi[t_idx] += U @ u_acc

            for near in [*node.near_list, node]:
                ndata = leaf_cache.get(near.box_id, {})
                if ndata.get('src_ps') is None:
                    continue
                K_near = kern.matrix(tgt_sub, ndata['src_ps'])
                phi[t_idx] += K_near @ q[near.src_idx]

        return phi

    def _solve_wideband(self, tree: QuadTree, X: PointSet, Y: PointSet,
                        q: np.ndarray) -> np.ndarray:
        """Wideband leaf-only FMM [HK §4 Algorithm 4.1].

        Selects the LF or HF basis per leaf based on the k*delta criterion
        [HK §4.2]:  LF if k*delta <= r/e,  HF otherwise.

        Same-regime pairs use the matching M2L translation matrix.
        Cross-regime pairs fall back to direct P2P (always correct).
        """
        src_pts = Y.points
        tgt_pts = X.points
        phi = np.zeros(len(tgt_pts), dtype=complex)
        kern = HelmholtzKernel(self.k)
        n2r1 = 2 * self.r + 1

        leaf_cache: Dict[int, dict] = {}
        for node in tree.postorder():
            if not node.is_leaf():
                continue
            s_idx = node.src_idx
            if len(s_idx) == 0:
                leaf_cache[node.box_id] = {
                    'v_lf': None, 'v_hf': None,
                    'src_ps': None, 'sf': None, 'regime': 'lf'}
                continue
            src_sub = PointSet(src_pts[s_idx])
            delta_s = max(src_sub.radius, 1e-15)
            reg = self._lm.regime(delta_s)
            if reg == 'lf':
                sf_s = ScalingFactors(self.r, self.k, delta_s)
                V = self._lm.build_basis_lf(src_sub, sf_s, self.balanced)
                leaf_cache[node.box_id] = {
                    'v_lf': V.T @ q[s_idx], 'v_hf': None,
                    'src_ps': src_sub, 'sf': sf_s, 'regime': 'lf'}
            else:
                V = self._lm.build_basis_hf(src_sub, sign=+1)
                leaf_cache[node.box_id] = {
                    'v_lf': None, 'v_hf': V.T @ q[s_idx],
                    'src_ps': src_sub, 'sf': None, 'regime': 'hf'}

        for node in tree.postorder():
            if not node.is_leaf():
                continue
            t_idx = node.tgt_idx
            if len(t_idx) == 0:
                continue
            tgt_sub = PointSet(tgt_pts[t_idx])
            delta_t = max(tgt_sub.radius, 1e-15)
            reg_t = self._lm.regime(delta_t)
            if reg_t == 'lf':
                sf_t = ScalingFactors(self.r, self.k, delta_t)
                U = self._lm.build_basis_lf(tgt_sub, sf_t, self.balanced)
            else:
                sf_t = None
                U = self._lm.build_basis_hf(tgt_sub, sign=-1)

            u_acc = np.zeros(n2r1, dtype=complex)
            for partner in node.interaction_list:
                pdata = leaf_cache.get(partner.box_id, {})
                src_sub_p = pdata.get('src_ps')
                if src_sub_p is None:
                    continue
                reg_p = pdata['regime']
                if reg_t == 'lf' and reg_p == 'lf':
                    B = self._lm.build_B_lf(
                        tgt_sub, src_sub_p, sf_t, pdata['sf'], self.balanced)
                    u_acc += B @ pdata['v_lf']
                elif reg_t == 'hf' and reg_p == 'hf':
                    B = self._lm.build_B_hf(tgt_sub, src_sub_p)
                    u_acc += B @ pdata['v_hf']
                else:
                    # Cross-regime: direct P2P fallback (always correct)
                    phi[t_idx] += kern.matrix(tgt_sub, src_sub_p) @ q[partner.src_idx]
            phi[t_idx] += U @ u_acc

            for near in [*node.near_list, node]:
                ndata = leaf_cache.get(near.box_id, {})
                if ndata.get('src_ps') is None:
                    continue
                phi[t_idx] += kern.matrix(tgt_sub, ndata['src_ps']) @ q[near.src_idx]

        return phi
