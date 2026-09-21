"""Compare the approximation with an exact compiled PG(1, eta) sampler."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import polyagamma
from polyagamma import random_polyagamma
from scipy.stats import ks_2samp, wasserstein_distance

from fast_polya_gamma_jax import pg1_mean, pg1_variance, sample_pg1


def _summarize(exact, approximate, eta, num_terms):
    exact_mean = float(np.mean(exact))
    approximate_mean = float(np.mean(approximate))
    exact_variance = float(np.var(exact))
    approximate_variance = float(np.var(approximate))
    theoretical_mean = float(pg1_mean(eta))
    theoretical_variance = float(pg1_variance(eta))
    quantile_levels = np.array([0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
    exact_quantiles = np.quantile(exact, quantile_levels)
    approximate_quantiles = np.quantile(approximate, quantile_levels)
    ks = ks_2samp(exact, approximate, method="asymp")
    return {
        "eta": eta,
        "num_terms": num_terms,
        "theoretical_mean": theoretical_mean,
        "exact_sample_mean": exact_mean,
        "approximate_sample_mean": approximate_mean,
        "approximate_mean_relative_error": (
            approximate_mean - theoretical_mean
        )
        / theoretical_mean,
        "theoretical_variance": theoretical_variance,
        "exact_sample_variance": exact_variance,
        "approximate_sample_variance": approximate_variance,
        "approximate_variance_relative_error": (
            approximate_variance - theoretical_variance
        )
        / theoretical_variance,
        "ks_statistic": float(ks.statistic),
        "wasserstein_distance": float(wasserstein_distance(exact, approximate)),
        "wasserstein_over_mean": float(
            wasserstein_distance(exact, approximate) / theoretical_mean
        ),
        "quantiles": [
            {
                "probability": float(probability),
                "exact": float(exact_value),
                "approximate": float(approximate_value),
                "relative_error": float(
                    (approximate_value - exact_value) / exact_value
                ),
            }
            for probability, exact_value, approximate_value in zip(
                quantile_levels, exact_quantiles, approximate_quantiles
            )
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument(
        "--etas",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 4.0, 8.0, 12.0],
    )
    parser.add_argument("--terms", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(20260921)
    key = jax.random.key(20260921)
    results = {
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "polyagamma_version": polyagamma.__version__,
        "samples_per_configuration": args.samples,
        "reference": "polyagamma.random_polyagamma(method='devroye')",
        "configurations": [],
    }
    for eta_index, eta in enumerate(args.etas):
        exact = random_polyagamma(
            h=1.0,
            z=eta,
            size=args.samples,
            method="devroye",
            random_state=rng,
        )
        eta_array = jnp.full((args.samples,), eta, dtype=jnp.float64)
        for num_terms in args.terms:
            sampler = jax.jit(
                partial(
                    sample_pg1,
                    num_terms=num_terms,
                    tail_correction=True,
                )
            )
            approximate = np.asarray(
                sampler(
                    jax.random.fold_in(key, eta_index * 1000 + num_terms),
                    eta_array,
                )
            )
            results["configurations"].append(
                _summarize(exact, approximate, eta, num_terms)
            )

    payload = json.dumps(results, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main()
