from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fast_polya_gamma_jax import pg1_mean, pg1_series_variance, sample_pg1

jax.config.update("jax_enable_x64", True)


def test_shape_positivity_and_reproducibility():
    eta = jnp.array([[0.0, 1.0], [-2.0, 6.0]])
    key = jax.random.key(17)
    first = sample_pg1(key, eta, num_terms=16)
    second = sample_pg1(key, eta, num_terms=16)
    assert first.shape == eta.shape
    assert bool(jnp.all(first > 0.0))
    np.testing.assert_array_equal(first, second)


def test_sampler_jits():
    sampler = jax.jit(partial(sample_pg1, num_terms=8, tail_correction=True))
    eta = jnp.linspace(-5.0, 5.0, 1024)
    draws = sampler(jax.random.key(0), eta)
    assert draws.shape == eta.shape
    assert bool(jnp.all(jnp.isfinite(draws)))


@pytest.mark.parametrize("eta_value", [0.0, 1.0, 4.0, 8.0])
def test_empirical_mean_matches_exact_mean(eta_value):
    n = 200_000
    eta = jnp.full((n,), eta_value, dtype=jnp.float64)
    draws = jax.jit(partial(sample_pg1, num_terms=16))(
        jax.random.key(int(eta_value * 10) + 1), eta
    )
    observed = float(jnp.mean(draws))
    expected = float(pg1_mean(eta_value))
    standard_error = float(jnp.sqrt(pg1_series_variance(eta_value, 16) / n))
    assert abs(observed - expected) < 5.0 * standard_error


def test_invalid_number_of_terms():
    with pytest.raises(ValueError, match="positive"):
        sample_pg1(jax.random.key(0), jnp.array([0.0]), num_terms=0)
