import numpy as np
import pytest

from stablefmmpy import PointSet, ScalingFactors, HelmholtzKernel


class TestPointSet:
    def test_center_is_mean(self):
        pts = np.array([0+0j, 2+0j, 1+0j])
        ps = PointSet(pts)
        assert abs(ps.center - 1+0j) < 1e-14

    def test_explicit_center(self):
        pts = np.array([0+0j, 2+0j])
        ps = PointSet(pts, center=5+0j)
        assert ps.center == 5+0j

    def test_radius_max_deviation(self):
        pts = np.array([0+0j, 3+0j])
        ps = PointSet(pts)
        assert abs(ps.radius - 1.5) < 1e-14

    def test_len(self):
        ps = PointSet.random_uniform(30, center=0+0j, radius=1.0, rng=0)
        assert len(ps) == 30

    def test_random_uniform_all_within_radius(self):
        rng = np.random.default_rng(7)
        ps = PointSet.random_uniform(200, center=0.5+0.5j, radius=0.3, rng=rng)
        dists = np.abs(ps.points - ps.center)
        assert np.all(dists <= ps.radius + 1e-12)
        assert ps.radius <= 0.3 + 1e-12

    def test_scaled(self):
        ps = PointSet(np.array([1+0j, 0+1j]))
        ps2 = ps.scaled(2.0)
        assert abs(ps2.radius - 2 * ps.radius) < 1e-14


class TestScalingFactors:
    def test_p0_is_one(self):
        sf = ScalingFactors(r=10, k=100.0, delta=0.001)
        assert abs(sf[0] - 1.0) < 1e-12

    def test_log_array_length(self):
        r = 20
        sf = ScalingFactors(r=r, k=10.0, delta=0.05)
        assert len(sf.log_array()) == r + 1

    def test_lambda_ge_one(self):
        sf = ScalingFactors(r=30, k=100.0, delta=0.001)
        arr = sf.as_array()
        assert np.all(arr >= 1.0 - 1e-10)

    def test_large_r_no_overflow(self):
        sf = ScalingFactors(r=200, k=100.0, delta=0.001)
        log_arr = sf.log_array()
        assert np.all(np.isfinite(log_arr))

    def test_ratio(self):
        sf = ScalingFactors(r=10, k=50.0, delta=0.01)
        ratio = sf.ratio(0, 0)
        assert abs(ratio - 1.0) < 1e-12


class TestHelmholtzKernel:
    def test_matrix_shape(self, separated_pair):
        X, Y, _ = separated_pair
        kern = HelmholtzKernel(k=10.0)
        K = kern.matrix(X, Y)
        assert K.shape == (len(X), len(Y))

    def test_matvec_shape(self, separated_pair):
        X, Y, q = separated_pair
        kern = HelmholtzKernel(k=10.0)
        phi = kern.matvec(X, Y, q)
        assert phi.shape == (len(X),)

    def test_relative_error_zero(self, separated_pair):
        X, Y, q = separated_pair
        kern = HelmholtzKernel(k=10.0)
        phi = kern.matvec(X, Y, q)
        err = kern.relative_error(phi, phi)
        assert err == 0.0

    def test_relative_error_is_instance_method(self, separated_pair):
        X, Y, q = separated_pair
        kern = HelmholtzKernel(k=10.0)
        phi = kern.matvec(X, Y, q)
        perturbed = phi + 0.001 * phi
        err = kern.relative_error(perturbed, phi)
        assert abs(err - 0.001) < 1e-10

    def test_cauchy_matrix_shape(self, separated_pair):
        X, Y, _ = separated_pair
        K = HelmholtzKernel.cauchy_matrix(X, Y)
        assert K.shape == (len(X), len(Y))

    def test_log_matrix_shape(self, separated_pair):
        X, Y, _ = separated_pair
        K = HelmholtzKernel.log_matrix(X, Y)
        assert K.shape == (len(X), len(Y))
