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

    Three internal modes:
    - _solve_single():     Multipole2D Algorithm 4.1 (single-regime FMM, leaf-only)
    - _solve_wideband():   HelmholtzKernel2D Algorithm 4.1 (LF + HF, leaf-only)
    - _solve_multilevel(): True multi-level FMM using M2M / M2L@all_levels / L2L,
                           implementing [M2D] Algorithm 4.1 and [HK] Algorithm 4.1
                           with the build_R_lf / build_R_hf restriction matrices.

    solve() dispatches to _solve_multilevel() when multilevel=True,
    otherwise to _solve_wideband() (default, backward-compatible behaviour).

    References:
      [M2D §4 Algorithm 4.1]  — single regime (non-oscillating kernels)
      [HK §4 Algorithm 4.1]   — wideband (LF + HF regimes)
      [HK §3 Theorem 3.1]     — inter-level restriction matrix R
    """

    def __init__(self, k: float, r: int, tau: float = 0.6, N0: int = 32,
                 balanced: bool = True, multilevel: bool = False):
        self.k = float(k)
        self.r = int(r)
        self.tau = tau
        self.N0 = N0
        self.balanced = balanced
        self.multilevel = multilevel
        self._lm = LeafMatrices(r, k)

    def solve(self, X: PointSet, Y: PointSet, q: np.ndarray) -> np.ndarray:
        """Compute phi = K q approximately via adaptive FMM. Builds tree internally."""
        tree = QuadTree()
        tree.build(Y.points, X.points, tau=self.tau, N0=self.N0,
                   multilevel=self.multilevel)
        if self.multilevel:
            return self._solve_multilevel(tree, X, Y, q)
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

    def _solve_multilevel(self, tree: QuadTree, X: PointSet, Y: PointSet,
                          q: np.ndarray) -> np.ndarray:
        """True multi-level FMM: M2M → M2L@all_levels → L2L → P2P.

        Implements [M2D] Algorithm 4.1 / [HK] Algorithm 4.1 using the
        inter-level restriction matrices from [HK §3 Theorem 3.1]:

          Phase 1 — M2M upward (postorder):
            leaf:     v = V(src_box, sf_box).T @ q[src_idx]
            interior: v_parent += R(child→parent) @ v_child
          Phase 2 — M2L at every level:
            t_node += sum_{B in interaction_list} B(node, B) @ v_B
          Phase 3 — L2L downward (preorder):
            u_child = R(parent→child) @ u_parent + t_child
          Phase 4 — leaf evaluation + P2P:
            phi[t] += U(tgt_box, sf_box) @ u_leaf
            phi[t] += K_exact(tgt, src) @ q[src]  for near-field pairs

        Key: V and U are built with PointSet(pts, center=node.center, radius=node.radius)
        so all bases are relative to the box centre, not the point-cloud centroid.
        This ensures R matrices (which use node.center displacements) compose correctly.

        The LF regime (k·radius ≤ r/e) uses build_R_lf / build_B_lf / build_basis_lf.
        HF regime leaves fall back to P2P (safe, always correct) to keep Phase 1–3
        clean and single-regime; a full wideband multi-level extension would wire in
        build_R_hf / build_B_hf at the HF leaf level.

        References: [M2D §4 Algorithm 4.1]; [HK §3 Theorem 3.1; §4 Algorithm 4.1]
        """
        src_pts = Y.points
        tgt_pts = X.points
        n2r1 = 2 * self.r + 1
        phi = np.zeros(len(tgt_pts), dtype=complex)
        kern = HelmholtzKernel(self.k)
        bal = self.balanced

        def box_ps(node: FMMNode) -> PointSet:
            """Single-point PointSet pinned to node's box centre/radius."""
            return PointSet(np.array([node.center]),
                            center=node.center, radius=node.radius)

        def sf(node: FMMNode) -> ScalingFactors:
            return ScalingFactors(self.r, self.k, max(node.radius, 1e-15))

        # ── Initialise all v / t / u vectors to zero ─────────────────────────
        for node in tree._nodes:
            node.v_vec = np.zeros(n2r1, dtype=complex)
            node.t_vec = np.zeros(n2r1, dtype=complex)
            node.u_vec = np.zeros(n2r1, dtype=complex)

        # ── Phase 1: M2M upward (postorder — children before parents) ────────
        # build_R_lf(child_ps, parent_ps, sf_child, sf_parent) computes:
        #   R[p, l] = (λ_parent[p] / λ_child[l]) · J_{p-l}(k|d|) · e^{i(p-l)·θ}
        # with d = child.center - parent.center.
        # v_parent = R @ v_child  translates the source multipole upward
        # via the Bessel addition theorem [HK §3 Theorem 3.1].
        for node in tree.postorder():
            if node.is_leaf():
                s = node.src_idx
                if len(s) == 0:
                    continue
                reg = self._lm.regime(max(node.radius, 1e-15))
                if reg == 'lf':
                    src_sub = PointSet(src_pts[s],
                                       center=node.center, radius=node.radius)
                    V = self._lm.build_basis_lf(src_sub, sf(node), bal)
                    node.v_vec = V.T @ q[s]
                # HF leaves: v_vec stays zero; contribution recovered in P2P (Phase 4)
            else:
                sf_node = sf(node)
                bps_node = box_ps(node)
                for child in node.children:
                    if not np.any(child.v_vec):
                        continue
                    # R[p,l] = (λ_parent[p]/λ_child[l])·J_{p-l}·phase  → M2M upward
                    R = self._lm.build_R_lf(
                        box_ps(child), bps_node, sf(child), sf_node, bal)
                    node.v_vec += R @ child.v_vec

        # ── Phase 2: M2L at every level ───────────────────────────────────────
        # Accumulates t_vec from all partners in interaction_list (built by
        # _build_multilevel_lists so that each pair appears at the coarsest level
        # where they first become well-separated — no double-counting).
        for node in tree._nodes:
            if not node.interaction_list:
                continue
            sf_t = sf(node)
            bps_t = box_ps(node)
            for partner in node.interaction_list:
                if not np.any(partner.v_vec):
                    continue
                B = self._lm.build_B_lf(
                    bps_t, box_ps(partner), sf_t, sf(partner), bal)
                node.t_vec += B @ partner.v_vec

        # ── Phase 3: L2L downward (preorder — parents before children) ───────
        # L2L uses the SAME R matrix as M2M but TRANSPOSED:
        #   R = build_R_lf(child_ps, parent_ps, sf_child, sf_parent)  [same as M2M]
        #   u_child = R.T @ u_parent + t_child
        #
        # Derivation (Bessel addition theorem, Graf's theorem):
        #   U_parent[i,p] = J_{|p|}(k|x-c_parent|)·λ_parent[p]·e^{ip·arg(x-c_parent)}
        #   Expanding x-c_parent = (x-c_child) + d  (d = c_child - c_parent):
        #   J_p(k|x-c_parent|)·e^{ip·arg} = Σ_l J_{p-l}(k|d|)·e^{i(p-l)·arg(d)} · J_l(k|x-c_child|)·e^{il·arg}
        #   → u_child[l] = Σ_p (λ_parent[p]/λ_child[l])·J_{p-l}(k|d|)·e^{i(p-l)θ_d}·u_parent[p]
        #                = (R_code.T @ u_parent)[l]   [HK §3 Theorem 3.1]
        #
        # Note: .T (plain transpose), NOT .conj().T, because J and λ are real
        # and the complex phase must NOT be conjugated.
        for node in tree.preorder():
            if node.parent is None:
                # Root: no parent → u_vec stays zero (root never has M2L)
                node.u_vec = node.t_vec.copy()
            else:
                # Same call as M2M (child first, parent second), but apply .T
                R = self._lm.build_R_lf(
                    box_ps(node), box_ps(node.parent),
                    sf(node), sf(node.parent), bal)
                node.u_vec = R.T @ node.parent.u_vec + node.t_vec

        # ── Phase 4: Leaf evaluation + P2P ───────────────────────────────────
        for node in tree.postorder():
            if not node.is_leaf():
                continue
            t = node.tgt_idx
            if len(t) == 0:
                continue

            tgt_sub = PointSet(tgt_pts[t], center=node.center, radius=node.radius)
            reg_t = self._lm.regime(max(node.radius, 1e-15))

            # Apply local expansion accumulated in u_vec
            if reg_t == 'lf' and np.any(node.u_vec):
                U = self._lm.build_basis_lf(tgt_sub, sf(node), bal)
                phi[t] += U @ node.u_vec

            # P2P: exact kernel for near-field pairs (near_list + self)
            for near in [*node.near_list, node]:
                s = near.src_idx
                if len(s) == 0:
                    continue
                src_actual = PointSet(src_pts[s])
                tgt_actual = PointSet(tgt_pts[t])
                phi[t] += kern.matrix(tgt_actual, src_actual) @ q[s]

            # HF-regime source leaves skipped in Phase 1 (v_vec=0) need a
            # direct P2P correction for their interaction_list partners.
            for partner in node.interaction_list:
                s = partner.src_idx
                if len(s) == 0:
                    continue
                if partner.is_leaf():
                    reg_s = self._lm.regime(max(partner.radius, 1e-15))
                    if reg_s == 'hf':
                        src_actual = PointSet(src_pts[s])
                        tgt_actual = PointSet(tgt_pts[t])
                        phi[t] += kern.matrix(tgt_actual, src_actual) @ q[s]

        return phi
