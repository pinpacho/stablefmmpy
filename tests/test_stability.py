import numpy as np
import pytest

from stablefmmpy import PointSet, StabilityAnalyzer, LeafMatrices, ScalingFactors


@pytest.fixture
def lf_well_separated():
    """LF regime, well-separated pair (k=10, delta=0.01)."""
    rng = np.random.default_rng(99)
    X = PointSet.random_uniform(40, center=0+0j, radius=0.01, rng=rng)
    Y = PointSet.random_uniform(40, center=0.08+0j, radius=0.01, rng=rng)
    return X, Y


class TestVerifyNormBounds:
    def test_theorem_satisfied(self, lf_well_separated):
        X, Y = lf_well_separated
        sa = StabilityAnalyzer(k=10.0, r=15)
        result = sa.verify_norm_bounds(X, Y)
        assert result['theorem_satisfied'] is True

    def test_u_max_le_one(self, lf_well_separated):
        X, Y = lf_well_separated
        sa = StabilityAnalyzer(k=10.0, r=15)
        result = sa.verify_norm_bounds(X, Y)
        assert result['U_max_balanced'] <= 1.0 + 1e-10

    def test_balanced_b_smaller_than_naive_b(self, lf_well_separated):
        X, Y = lf_well_separated
        sa = StabilityAnalyzer(k=10.0, r=15)
        result = sa.verify_norm_bounds(X, Y)
        assert result['B_max_balanced'] < result['B_max_naive']


class TestSweepRank:
    def test_error_decreases_with_rank(self, lf_well_separated):
        X, Y = lf_well_separated
        rng = np.random.default_rng(11)
        q = rng.standard_normal(len(X)) + 1j * rng.standard_normal(len(X))
        sa = StabilityAnalyzer(k=10.0, r=20)
        rows = sa.sweep_rank(X, Y, q, r_values=[5, 10, 15, 20])
        errors = [row['stable_err'] for row in rows]
        # Errors should be generally decreasing as r increases
        assert errors[-1] < errors[0]

    def test_rows_have_expected_keys(self, lf_well_separated):
        X, Y = lf_well_separated
        rng = np.random.default_rng(12)
        q = rng.standard_normal(len(X)) + 1j * rng.standard_normal(len(X))
        sa = StabilityAnalyzer(k=10.0, r=10)
        rows = sa.sweep_rank(X, Y, q, r_values=[5, 10])
        assert len(rows) == 2
        for row in rows:
            assert set(row.keys()) == {'r', 'stable_B', 'stable_err', 'naive_B', 'naive_err'}


class TestFindOverflowThreshold:
    def test_returns_int(self, lf_well_separated):
        X, Y = lf_well_separated
        sa = StabilityAnalyzer(k=100.0, r=50)
        threshold = sa.find_overflow_threshold(X, Y, r_max=60, step=10)
        assert isinstance(threshold, int)
