import numpy as np
import pytest

from stablefmmpy import PointSet, ScalingFactors, LeafMatrices


@pytest.fixture
def lf_pair():
    """LF regime: k=10, delta=0.01 -> k*delta=0.1, r/e≈7.36, 0.1 < 7.36 -> LF."""
    rng = np.random.default_rng(0)
    X = PointSet.random_uniform(30, center=0+0j, radius=0.01, rng=rng)
    Y = PointSet.random_uniform(30, center=0.08+0j, radius=0.01, rng=rng)
    return X, Y


class TestRegime:
    def test_lf(self):
        lm = LeafMatrices(r=20, k=10.0)
        assert lm.regime(delta=0.01) == 'lf'   # k*delta=0.1 <= 20/e≈7.36

    def test_hf(self):
        lm = LeafMatrices(r=20, k=100.0)
        assert lm.regime(delta=1.0) == 'hf'    # k*delta=100 > 20/e≈7.36

    def test_boundary(self):
        lm = LeafMatrices(r=20, k=1.0)
        # At k*delta = r/e exactly: LF (<=)
        delta_exact = 20.0 / np.e
        assert lm.regime(delta=delta_exact) == 'lf'


class TestBuildBasisLF:
    def test_shape(self, lf_pair):
        X, Y = lf_pair
        lm = LeafMatrices(r=10, k=10.0)
        sf = ScalingFactors(r=10, k=10.0, delta=X.radius)
        U = lm.build_basis_lf(X, sf, balanced=True)
        assert U.shape == (len(X), 2 * 10 + 1)

    def test_balanced_max_le_one(self, lf_pair):
        X, Y = lf_pair
        lm = LeafMatrices(r=15, k=10.0)
        sf = ScalingFactors(r=15, k=10.0, delta=X.radius)
        U = lm.build_basis_lf(X, sf, balanced=True)
        assert np.max(np.abs(U)) <= 1.0 + 1e-10

    def test_balanced_less_than_naive_max(self, lf_pair):
        X, Y = lf_pair
        r = 15
        lm = LeafMatrices(r=r, k=10.0)
        sf_bal = ScalingFactors(r=r, k=10.0, delta=X.radius)
        sf_zero = ScalingFactors.__new__(ScalingFactors)
        sf_zero.r, sf_zero.k, sf_zero.delta = r, 10.0, X.radius
        sf_zero._log_lam = np.zeros(r + 1)
        U_bal = lm.build_basis_lf(X, sf_bal, balanced=True)
        U_naive = lm.build_basis_lf(X, sf_zero, balanced=False)
        assert np.max(np.abs(U_bal)) <= np.max(np.abs(U_naive)) + 1e-10


class TestBuildBasisHF:
    def test_shape(self):
        rng = np.random.default_rng(1)
        X = PointSet.random_uniform(20, center=0+0j, radius=2.0, rng=rng)
        lm = LeafMatrices(r=10, k=100.0)
        U = lm.build_basis_hf(X, sign=-1)
        assert U.shape == (len(X), 2 * 10 + 1)

    def test_all_finite(self):
        rng = np.random.default_rng(2)
        X = PointSet.random_uniform(20, center=0+0j, radius=5.0, rng=rng)
        lm = LeafMatrices(r=30, k=500.0)
        U = lm.build_basis_hf(X, sign=-1)
        assert np.all(np.isfinite(U))


class TestBalanceFactor:
    def test_balanced_b_finite(self, lf_pair):
        X, Y = lf_pair
        lm = LeafMatrices(r=15, k=10.0)
        sf_x = ScalingFactors(r=15, k=10.0, delta=X.radius)
        sf_y = ScalingFactors(r=15, k=10.0, delta=Y.radius)
        B_bal = lm.build_B_lf(X, Y, sf_x, sf_y, balanced=True)
        assert np.isfinite(lm.balance_factor(B_bal))

    def test_balanced_smaller_than_naive(self, lf_pair):
        X, Y = lf_pair
        r = 30
        lm = LeafMatrices(r=r, k=100.0)
        sf_x = ScalingFactors(r=r, k=100.0, delta=X.radius)
        sf_y = ScalingFactors(r=r, k=100.0, delta=Y.radius)
        sf0_x = ScalingFactors.__new__(ScalingFactors)
        sf0_x.r, sf0_x.k, sf0_x.delta = r, 100.0, X.radius
        sf0_x._log_lam = np.zeros(r + 1)
        sf0_y = ScalingFactors.__new__(ScalingFactors)
        sf0_y.r, sf0_y.k, sf0_y.delta = r, 100.0, Y.radius
        sf0_y._log_lam = np.zeros(r + 1)
        B_bal = lm.build_B_lf(X, Y, sf_x, sf_y, balanced=True)
        B_naive = lm.build_B_lf(X, Y, sf0_x, sf0_y, balanced=False)
        assert lm.balance_factor(B_bal) < lm.balance_factor(B_naive)


class TestFactorize:
    def test_factorize_lf_approximation(self, lf_pair):
        X, Y = lf_pair
        rng = np.random.default_rng(5)
        q = rng.standard_normal(len(Y)) + 1j * rng.standard_normal(len(Y))
        lm = LeafMatrices(r=20, k=10.0)
        U, B, V = lm.factorize(X, Y, balanced=True, regime='lf')
        phi_approx = U @ (B @ (V.T @ q))
        from stablefmmpy import HelmholtzKernel
        kern = HelmholtzKernel(k=10.0)
        phi_exact = kern.matvec(X, Y, q)
        err = kern.relative_error(phi_approx, phi_exact)
        assert err < 1e-6
