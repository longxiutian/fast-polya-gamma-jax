"""Analytical moments for PG(1, eta) and its finite-series approximation."""

from __future__ import annotations

import jax.numpy as jnp


def _as_float_array(eta):
    eta = jnp.asarray(eta)
    if not jnp.issubdtype(eta.dtype, jnp.inexact):
        eta = eta.astype(jnp.result_type(float))
    return eta


def pg1_mean(eta):
    """Return the exact mean of a PG(1, eta) random variable."""
    eta = _as_float_array(eta)
    absolute_eta = jnp.abs(eta)
    eta2 = absolute_eta * absolute_eta
    series = 0.25 - eta2 / 48.0 + eta2 * eta2 / 480.0
    safe_eta = jnp.where(absolute_eta < 1e-3, 1.0, absolute_eta)
    ratio = jnp.tanh(safe_eta / 2.0) / (2.0 * safe_eta)
    return jnp.where(absolute_eta < 1e-3, series, ratio)


def pg1_variance(eta):
    """Return the exact variance of a PG(1, eta) random variable."""
    eta = _as_float_array(eta)
    absolute_eta = jnp.abs(eta)
    eta2 = absolute_eta * absolute_eta
    series = (
        1.0 / 24.0
        - eta2 / 120.0
        + 17.0 * eta2 * eta2 / 13440.0
        - 31.0 * eta2 * eta2 * eta2 / 181440.0
    )
    safe_eta = jnp.where(absolute_eta < 1e-2, 1.0, absolute_eta)
    tanh_half = jnp.tanh(safe_eta / 2.0)
    ratio = (
        tanh_half - 0.5 * safe_eta * (1.0 - jnp.square(tanh_half))
    ) / (
        2.0 * safe_eta**3
    )
    return jnp.where(absolute_eta < 1e-2, series, ratio)


def _series_denominators(eta, num_terms: int):
    eta = _as_float_array(eta)
    k = jnp.arange(num_terms, dtype=eta.dtype) + 0.5
    tilt = jnp.square(eta[..., None] / (2.0 * jnp.pi))
    return jnp.square(k) + tilt


def pg1_series_mean(eta, num_terms: int):
    """Mean of the first ``num_terms`` in the PG(1, eta) gamma series."""
    denominator = _series_denominators(eta, num_terms)
    return jnp.sum(jnp.reciprocal(denominator), axis=-1) / (2.0 * jnp.pi**2)


def pg1_series_variance(eta, num_terms: int):
    """Variance of the first ``num_terms`` in the PG(1, eta) gamma series."""
    denominator = _series_denominators(eta, num_terms)
    return jnp.sum(jnp.reciprocal(jnp.square(denominator)), axis=-1) / (
        4.0 * jnp.pi**4
    )


def pg1_tail_mean(eta, num_terms: int):
    """Expected value of the omitted infinite-series tail."""
    return jnp.maximum(pg1_mean(eta) - pg1_series_mean(eta, num_terms), 0.0)


def pg1_tail_variance(eta, num_terms: int):
    """Variance of the omitted infinite-series tail."""
    return jnp.maximum(
        pg1_variance(eta) - pg1_series_variance(eta, num_terms), 0.0
    )
