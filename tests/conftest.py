import numpy as np
import pytest

from stablefmmpy import PointSet


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def separated_pair(rng):
    """Two well-separated point sets in the LF regime (k=10, delta=0.01, k*delta=0.1 << r/e)."""
    X = PointSet.random_uniform(50, center=0+0j, radius=0.01, rng=rng)
    Y = PointSet.random_uniform(50, center=0.1+0j, radius=0.01, rng=rng)
    q = rng.standard_normal(50) + 1j * rng.standard_normal(50)
    return X, Y, q
