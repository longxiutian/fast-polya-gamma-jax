"""NumPyro integration for the pure-JAX PG(1, eta) sampler."""

from __future__ import annotations

from collections.abc import Callable

from .sampler import sample_pg1


def make_pg1_gibbs_fn(
    linear_predictor_fn: Callable,
    *,
    site_name: str = "omega",
    num_terms: int = 16,
    tail_correction: bool = True,
):
    """Build an update callable accepted by NumPyro ``CustomGibbs``.

    ``linear_predictor_fn`` is called with keyword arguments
    ``gibbs_sites`` and ``hmc_sites`` and must return the current ``eta``
    array. NumPyro retains the historical name ``hmc_sites`` for all sites
    conditioning a custom Gibbs block, even when no HMC block is present.
    """
    if not callable(linear_predictor_fn):
        raise TypeError("linear_predictor_fn must be callable")

    def gibbs_fn(*, rng_key, gibbs_sites, hmc_sites):
        eta = linear_predictor_fn(
            gibbs_sites=gibbs_sites,
            hmc_sites=hmc_sites,
        )
        omega = sample_pg1(
            rng_key,
            eta,
            num_terms=num_terms,
            tail_correction=tail_correction,
        )
        return {site_name: omega}

    return gibbs_fn


def make_custom_gibbs(*args, **kwargs):
    """Return a NumPyro ``CustomGibbs`` block for the approximate sampler."""
    try:
        from numpyro.infer import CustomGibbs
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        raise ImportError(
            "NumPyro integration requires `pip install fast-polya-gamma-jax[numpyro]`."
        ) from exc
    return CustomGibbs(make_pg1_gibbs_fn(*args, **kwargs))
