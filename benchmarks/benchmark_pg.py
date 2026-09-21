"""Benchmark PG sampling kernels after compilation."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp

from fast_polya_gamma_jax import sample_pg1


def _time_kernel(kernel, key, eta, repeats):
    compiled = kernel.lower(key, eta).compile()
    compiled(key, eta).block_until_ready()
    timings = []
    for index in range(repeats):
        run_key = jax.random.fold_in(key, index)
        started = time.perf_counter()
        compiled(run_key, eta).block_until_ready()
        timings.append(time.perf_counter() - started)
    return {
        "median_seconds": statistics.median(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=60_000)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    eta = jnp.linspace(-8.0, 8.0, args.size, dtype=jnp.float64)
    key = jax.random.key(20260921)
    kernels = {
        "custom_k8_tail": jax.jit(
            partial(sample_pg1, num_terms=8, tail_correction=True)
        ),
        "custom_k16_tail": jax.jit(
            partial(sample_pg1, num_terms=16, tail_correction=True)
        ),
    }
    numpyro_version = None

    try:
        import numpyro
        import numpyro.distributions as dist

        numpyro_version = numpyro.__version__
        kernels["numpyro_truncated_k8"] = jax.jit(
            lambda sample_key, values: dist.TruncatedPolyaGamma(
                batch_shape=values.shape
            ).sample(sample_key)
        )
    except ImportError:
        pass

    results = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "dtype": str(eta.dtype),
        "jax_version": jax.__version__,
        "numpyro_version": numpyro_version,
        "size": args.size,
        "repeats": args.repeats,
        "kernels": {},
    }
    for name, kernel in kernels.items():
        timing = _time_kernel(kernel, key, eta, args.repeats)
        timing["draws_per_second"] = args.size / timing["median_seconds"]
        results["kernels"][name] = timing

    payload = json.dumps(results, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main()
