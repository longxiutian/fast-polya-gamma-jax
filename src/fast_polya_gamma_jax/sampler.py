"""Fixed-cost JAX sampler for approximate PG(1, eta) random variables."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .moments import pg1_tail_mean


def sample_pg1(
    key: jax.Array,
    eta,
    *,
    num_terms: int = 16,
    tail_correction: bool = True,
):
    """Draw an approximate PG(1, eta) sample using a finite gamma series.

    Because Gamma(1, 1) equals Exponential(1), this implementation uses
    ``jax.random.exponential`` instead of a general rejection-based gamma
    sampler. ``num_terms`` must remain static when this function is JIT
    compiled.

    Args:
        key: JAX random key.
        eta: Scalar or array of tilting parameters.
        num_terms: Number of explicit infinite-series terms.
        tail_correction: Add the exact expected value of the omitted tail.

    Returns:
        An array with the same shape as ``eta``.
    """
    if num_terms < 1:
        raise ValueError("num_terms must be positive")

    eta = jnp.asarray(eta)
    if not jnp.issubdtype(eta.dtype, jnp.inexact):
        eta = eta.astype(jnp.result_type(float))

    k = jnp.arange(num_terms, dtype=eta.dtype) + 0.5
    denominator = jnp.square(k) + jnp.square(
        eta[..., None] / (2.0 * jnp.pi)
    )
    exponentials = jax.random.exponential(
        key,
        shape=eta.shape + (num_terms,),
        dtype=eta.dtype,
    )
    sample = jnp.sum(exponentials / denominator, axis=-1) / (2.0 * jnp.pi**2)

    if tail_correction:
        sample = sample + pg1_tail_mean(eta, num_terms)
    return sample
