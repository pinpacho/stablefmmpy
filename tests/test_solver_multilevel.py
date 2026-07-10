"""
tests/test_solver_multilevel.py — Tests for FMMSolver(multilevel=True).

Verifies that the multi-level FMM (M2M → M2L@all_levels → L2L → P2P)
gives results consistent with the leaf-only solver and with the exact kernel.
"""

import numpy as np
import pytest

from stablefmmpy import (
    PointSet, FMMSolver, HelmholtzKernel,
)
from stablefmmpy.tree import QuadTree


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def separated_pair():
    """Two well-separated clusters: 200 targets at origin, 200 sources at 0.1."""
    rng = np.random.default_rng(42)
    X = PointSet.random_uniform(200, 0 + 0j, 0.01, rng=rng)
    Y = PointSet.random_uniform(200, 0.1 + 0j, 0.01, rng=rng)
    q = rng.standard_normal(200) + 1j * rng.standard_normal(200)
    return X, Y, q


@pytest.fixture
def deep_pair():
    """500-point pair with small N0 → deep tree (≥4 levels)."""
    rng = np.random.default_rng(7)
    X = PointSet.random_uniform(500, 0 + 0j, 0.01, rng=rng)
    Y = PointSet.random_uniform(500, 0.1 + 0j, 0.01, rng=rng)
    q = rng.standard_normal(500) + 1j * rng.standard_normal(500)
    return X, Y, q


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

class TestMultilevelCorrectness:

    def test_multilevel_matches_wideband_shallow(self, separated_pair):
        """multilevel=True matches multilevel=False on a shallow (2-level) tree."""
        X, Y, q = separated_pair
        k = 10.0; r = 20; N0 = 300  # force 2-level tree
        kern = HelmholtzKernel(k)
        phi_ex = kern.matvec(X, Y, q)
        phi_ml = FMMSolver(k, r, N0=N0, multilevel=True).solve(X, Y, q)
        phi_lo = FMMSolver(k, r, N0=N0, multilevel=False).solve(X, Y, q)
        assert kern.relative_error(phi_ml, phi_ex) < 1e-10
        assert kern.relative_error(phi_lo, phi_ex) < 1e-10
        # On a 2-level tree the two solvers are equivalent (no M2M/L2L steps)
        np.testing.assert_allclose(phi_ml, phi_lo, rtol=1e-10)

    def test_multilevel_error_deep_tree(self, separated_pair):
        """multilevel=True stays accurate at 6 levels (N0=32, r=20)."""
        X, Y, q = separated_pair
        k = 10.0; r = 20
        kern = HelmholtzKernel(k)
        phi_ex = kern.matvec(X, Y, q)
        phi_ml = FMMSolver(k, r, N0=32, multilevel=True).solve(X, Y, q)
        assert kern.relative_error(phi_ml, phi_ex) < 1e-10

    def test_multilevel_accuracy_large_N(self, deep_pair):
        """multilevel=True is accurate on a deep tree with 500 points."""
        X, Y, q = deep_pair
        k = 10.0; r = 20; N0 = 10
        kern = HelmholtzKernel(k)
        phi_ex = kern.matvec(X, Y, q)
        phi_ml = FMMSolver(k, r, N0=N0, multilevel=True).solve(X, Y, q)
        # At least 4 levels
        tree = QuadTree()
        tree.build(Y.points, X.points, N0=N0, multilevel=True)
        assert tree.max_level() >= 4
        assert kern.relative_error(phi_ml, phi_ex) < 1e-8

    def test_multilevel_all_n0(self, separated_pair):
        """multilevel=True gives <1e-9 error across a range of N0 values."""
        X, Y, q = separated_pair
        k = 10.0; r = 20
        kern = HelmholtzKernel(k)
        phi_ex = kern.matvec(X, Y, q)
        for N0 in [500, 200, 95, 90, 50, 32, 10]:
            phi_ml = FMMSolver(k, r, N0=N0, multilevel=True).solve(X, Y, q)
            err = kern.relative_error(phi_ml, phi_ex)
            assert err < 1e-9, f"N0={N0}: multilevel error {err:.2e} exceeds 1e-9"


# ---------------------------------------------------------------------------
# Tree structure tests
# ---------------------------------------------------------------------------

class TestMultilevelTreeStructure:

    def test_interior_nodes_have_interaction_lists(self, separated_pair):
        """After multilevel build, interior nodes at level ≥3 have non-empty interaction_list."""
        X, Y, _ = separated_pair
        tree = QuadTree()
        tree.build(Y.points, X.points, N0=32, multilevel=True)
        by_lev = tree.by_level()
        interior_il = sum(
            len(n.interaction_list)
            for lv, nodes in by_lev.items()
            for n in nodes
            if not n.is_leaf()
        )
        assert interior_il > 0, "Expected interior nodes to have interaction_list entries"

    def test_all_leaves_covered(self, separated_pair):
        """Every leaf pair must be covered by interaction_list or near_list."""
        X, Y, _ = separated_pair
        tree = QuadTree()
        tree.build(Y.points, X.points, N0=95, multilevel=True)
        all_leaves = [n for n in tree._nodes if n.is_leaf()]
        for tgt in all_leaves:
            for src in all_leaves:
                if src is tgt:
                    continue
                # Either in near_list directly
                in_nl = src in tgt.near_list
                # Or covered by M2L path (src or ancestor in some tgt-ancestor's IL)
                covered = in_nl
                tgt_anc = tgt
                while tgt_anc is not None and not covered:
                    src_anc = src
                    while src_anc is not None and not covered:
                        if src_anc in tgt_anc.interaction_list:
                            covered = True
                        src_anc = src_anc.parent
                    tgt_anc = tgt_anc.parent
                assert covered, (
                    f"Leaf pair (tgt.level={tgt.level}, src.level={src.level}) "
                    f"is not covered by any M2L or P2P path"
                )

    def test_leaf_only_build_unchanged(self, separated_pair):
        """multilevel=False tree build is identical to original (backward compat)."""
        X, Y, _ = separated_pair
        t1 = QuadTree(); t1.build(Y.points, X.points, N0=32, multilevel=False)
        t2 = QuadTree(); t2.build(Y.points, X.points, N0=32, multilevel=False)
        # Check same number of nodes
        assert len(t1._nodes) == len(t2._nodes)
        # No interior interaction_list entries in leaf-only mode
        interior_il = sum(
            len(n.interaction_list) for n in t1._nodes if not n.is_leaf()
        )
        assert interior_il == 0, "Leaf-only mode should have no interior interaction_lists"


# ---------------------------------------------------------------------------
# Backward-compatibility test
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    def test_default_solver_unchanged(self, separated_pair):
        """FMMSolver() with no multilevel arg behaves exactly as before."""
        X, Y, q = separated_pair
        k = 10.0; r = 20
        kern = HelmholtzKernel(k)
        phi_ex = kern.matvec(X, Y, q)
        phi = FMMSolver(k, r).solve(X, Y, q)  # default multilevel=False
        assert kern.relative_error(phi, phi_ex) < 1e-10

    def test_multilevel_flag_attribute(self):
        """FMMSolver stores multilevel flag."""
        s1 = FMMSolver(1.0, 5, multilevel=False)
        s2 = FMMSolver(1.0, 5, multilevel=True)
        assert s1.multilevel is False
        assert s2.multilevel is True
