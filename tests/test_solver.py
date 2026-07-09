import numpy as np
import pytest

from stablefmmpy import FMMSolver, HelmholtzKernel, PointSet


def test_solve_returns_correct_shape(separated_pair):
    X, Y, q = separated_pair
    solver = FMMSolver(k=10.0, r=15, tau=0.6, N0=32, balanced=True)
    phi = solver.solve(X, Y, q)
    assert phi.shape == (len(X),)


def test_solve_relative_error_lf(separated_pair):
    """FMM approximation error < 1e-6 for LF regime (k=10, r=20)."""
    X, Y, q = separated_pair
    solver = FMMSolver(k=10.0, r=20, tau=0.6, N0=32, balanced=True)
    phi_fmm = solver.solve(X, Y, q)
    kern = HelmholtzKernel(10.0)
    phi_exact = kern.matvec(X, Y, q)
    err = kern.relative_error(phi_fmm, phi_exact)
    assert err < 1e-6


def test_solve_balanced_better_than_naive(separated_pair):
    """Balanced FMM should match exact much better than naive for large r."""
    X, Y, q = separated_pair
    k = 10.0
    r = 20
    kern = HelmholtzKernel(k)
    phi_exact = kern.matvec(X, Y, q)

    solver_bal = FMMSolver(k=k, r=r, tau=0.6, N0=32, balanced=True)
    phi_bal = solver_bal.solve(X, Y, q)
    err_bal = kern.relative_error(phi_bal, phi_exact)

    solver_naive = FMMSolver(k=k, r=r, tau=0.6, N0=32, balanced=False)
    phi_naive = solver_naive.solve(X, Y, q)
    err_naive = kern.relative_error(phi_naive, phi_exact)

    assert err_bal <= err_naive + 1e-8


def test_solve_high_k(rng):
    """HF regime smoke test: k=200, large delta -> HF basis selected; result is finite."""
    X = PointSet.random_uniform(30, center=0+0j, radius=0.5, rng=rng)
    Y = PointSet.random_uniform(30, center=3+0j, radius=0.5, rng=rng)
    q = rng.standard_normal(30) + 1j * rng.standard_normal(30)

    k = 200.0
    r = 15
    solver = FMMSolver(k=k, r=r, tau=0.6, N0=10, balanced=True)
    phi_fmm = solver.solve(X, Y, q)
    # HF with r=15 is low-rank; just verify output shape and finiteness.
    assert phi_fmm.shape == (len(X),)
    assert np.all(np.isfinite(phi_fmm))
