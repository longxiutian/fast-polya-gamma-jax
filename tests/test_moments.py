import jax
import jax.numpy as jnp
import numpy as np

from fast_polya_gamma_jax import (
    pg1_mean,
    pg1_series_mean,
    pg1_series_variance,
    pg1_tail_mean,
    pg1_tail_variance,
    pg1_variance,
)

jax.config.update("jax_enable_x64", True)


def test_exact_moments_at_zero():
    np.testing.assert_allclose(pg1_mean(0.0), 0.25, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(pg1_variance(0.0), 1.0 / 24.0, rtol=0.0, atol=1e-14)


def test_moments_are_even_in_eta():
    eta = jnp.array([0.1, 1.0, 4.0, 8.0])
    np.testing.assert_allclose(pg1_mean(eta), pg1_mean(-eta), rtol=1e-13)
    np.testing.assert_allclose(pg1_variance(eta), pg1_variance(-eta), rtol=1e-13)


def test_moments_remain_finite_for_large_tilts():
    eta = jnp.array([-1000.0, -100.0, 100.0, 1000.0])
    assert bool(jnp.all(jnp.isfinite(pg1_mean(eta))))
    assert bool(jnp.all(jnp.isfinite(pg1_variance(eta))))
    assert bool(jnp.all(pg1_mean(eta) > 0.0))
    assert bool(jnp.all(pg1_variance(eta) > 0.0))


def test_head_and_tail_reconstruct_exact_moments():
    eta = jnp.array([0.0, 0.5, 2.0, 6.0])
    for num_terms in (8, 16, 32):
        np.testing.assert_allclose(
            pg1_series_mean(eta, num_terms) + pg1_tail_mean(eta, num_terms),
            pg1_mean(eta),
            rtol=1e-13,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            pg1_series_variance(eta, num_terms)
            + pg1_tail_variance(eta, num_terms),
            pg1_variance(eta),
            rtol=1e-12,
            atol=1e-14,
        )


def test_tail_variance_is_small_for_k16():
    eta = jnp.linspace(-10.0, 10.0, 101)
    relative = pg1_tail_variance(eta, 16) / pg1_variance(eta)
    assert float(jnp.max(relative)) < 5e-4
