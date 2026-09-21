"""Benchmark a structurally faithful hierarchical block Gibbs workload.

The generated data are synthetic.  This harness is intended to test numerical
stability and computational scaling, not posterior recovery or substantive
interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from fast_polya_gamma_jax import (
    __version__,
    default_block_gibbs_priors,
    initialize_block_gibbs_state,
    prepare_block_gibbs_data,
    run_block_gibbs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people", type=int, default=20_000)
    parser.add_argument("--mean-observations", type=float, default=2.0)
    parser.add_argument("--max-observations", type=int, default=4)
    parser.add_argument("--campaigns", type=int, default=88)
    parser.add_argument("--items", type=int, default=340)
    parser.add_argument("--controls", type=int, default=22)
    parser.add_argument("--respondent-features", type=int, default=73)
    parser.add_argument("--warmup-sweeps", type=int, default=20)
    parser.add_argument("--sample-sweeps", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--num-terms", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    integer_fields = (
        "people",
        "max_observations",
        "campaigns",
        "items",
        "controls",
        "respondent_features",
        "warmup_sweeps",
        "sample_sweeps",
        "chunk_size",
        "num_terms",
    )
    for field in integer_fields:
        if getattr(args, field) < 1:
            raise ValueError(f"{field} must be positive")
    if not 1.0 <= args.mean_observations <= args.max_observations:
        raise ValueError("mean_observations must be between 1 and max_observations")
    if args.warmup_sweeps % args.chunk_size:
        raise ValueError("warmup_sweeps must be divisible by chunk_size")
    if args.sample_sweeps % args.chunk_size:
        raise ValueError("sample_sweeps must be divisible by chunk_size")


def _stable_expit(value: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -value))


def generate_synthetic_panel(args: argparse.Namespace, dtype: np.dtype):
    rng = np.random.default_rng(args.seed)
    extra_probability = (
        (args.mean_observations - 1.0) / (args.max_observations - 1.0)
        if args.max_observations > 1
        else 0.0
    )
    counts = 1 + rng.binomial(
        args.max_observations - 1, extra_probability, size=args.people
    )
    person_index = np.repeat(np.arange(args.people, dtype=np.int32), counts)
    observations = len(person_index)

    respondent_features = rng.normal(
        size=(args.people, args.respondent_features)
    ).astype(dtype)
    respondent_features -= respondent_features.mean(axis=0, keepdims=True)
    respondent_features /= respondent_features.std(axis=0, keepdims=True)
    hierarchy_design = np.concatenate(
        [np.ones((args.people, 1), dtype=dtype), respondent_features], axis=1
    )

    item_to_campaign = rng.integers(0, args.campaigns, size=args.items, dtype=np.int32)
    item_index = rng.integers(0, args.items, size=observations, dtype=np.int32)
    campaign_index = item_to_campaign[item_index]
    item_creative = rng.binomial(1, 0.5, size=(args.items, 3)).astype(dtype)
    creative_features = item_creative[item_index]
    controls = rng.normal(size=(observations, args.controls)).astype(dtype)

    true_hierarchy = rng.normal(
        0.0, 0.02, size=(args.respondent_features + 1, 3)
    ).astype(dtype)
    true_hierarchy[0] = np.asarray([0.25, -0.15, 0.10], dtype=dtype)
    true_sigma_b = np.asarray(
        [
            [0.1225, 0.02625, -0.00875],
            [0.02625, 0.09, 0.015],
            [-0.00875, 0.015, 0.0625],
        ],
        dtype=dtype,
    )
    true_b = hierarchy_design @ true_hierarchy
    true_b += (
        rng.normal(size=(args.people, 3)).astype(dtype)
        @ np.linalg.cholesky(true_sigma_b).T
    )
    true_a = rng.normal(-0.4, 0.35, size=args.people).astype(dtype)
    true_person = np.concatenate([true_a[:, None], true_b], axis=1)
    true_campaign = rng.normal(0.0, 0.15, size=args.campaigns).astype(dtype)
    true_item = rng.normal(0.0, 0.20, size=args.items).astype(dtype)
    true_delta = rng.normal(0.0, 0.08, size=args.controls).astype(dtype)
    person_design = np.concatenate(
        [np.ones((observations, 1), dtype=dtype), creative_features], axis=1
    )
    eta = (
        np.sum(person_design * true_person[person_index], axis=1)
        + true_campaign[campaign_index]
        + true_item[item_index]
        + controls @ true_delta
    )
    y = rng.binomial(1, _stable_expit(eta)).astype(dtype)
    return {
        "y": y,
        "person_index": person_index,
        "campaign_index": campaign_index,
        "item_index": item_index,
        "creative_features": creative_features,
        "controls": controls,
        "respondent_features": respondent_features,
        "counts": counts,
        "generated_eta": eta,
    }


def _metric_summary(array: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def _effective_sample_size(trace: np.ndarray) -> float:
    trace = np.asarray(trace, dtype=np.float64)
    size = len(trace)
    if size < 4 or not np.isfinite(trace).all():
        return float("nan")
    centered = trace - trace.mean()
    variance = np.dot(centered, centered) / size
    if variance <= 0.0:
        return float(size)
    autocorrelation = np.correlate(centered, centered, mode="full")[size - 1 :]
    autocorrelation /= variance * np.arange(size, 0, -1)
    paired_sum = 0.0
    for lag in range(1, size - 1, 2):
        pair = autocorrelation[lag] + autocorrelation[lag + 1]
        if pair <= 0.0:
            break
        paired_sum += pair
    return float(min(size, size / (1.0 + 2.0 * paired_sum)))


def _rss_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe_memory_stats(device) -> dict | None:
    stats = device.memory_stats()
    if stats is None:
        return None
    return {
        str(key): int(value) if isinstance(value, (int, np.integer)) else value
        for key, value in stats.items()
    }


def main() -> None:
    args = parse_args()
    _validate_args(args)
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = np.dtype(args.dtype)
    jax_dtype = jnp.float64 if args.dtype == "float64" else jnp.float32

    panel = generate_synthetic_panel(args, dtype)
    data = prepare_block_gibbs_data(
        y=panel["y"],
        person_index=panel["person_index"],
        campaign_index=panel["campaign_index"],
        item_index=panel["item_index"],
        creative_features=panel["creative_features"],
        controls=panel["controls"],
        respondent_features=panel["respondent_features"],
    )
    priors = default_block_gibbs_priors(dtype=jax_dtype)
    state = initialize_block_gibbs_state(
        jax.random.key(args.seed + 1),
        people=args.people,
        campaigns=args.campaigns,
        items=args.items,
        controls=args.controls,
        hierarchy_columns=args.respondent_features + 1,
        dtype=jax_dtype,
    )
    run_chunk = jax.jit(
        partial(
            run_block_gibbs,
            num_steps=args.chunk_size,
            num_terms=args.num_terms,
            tail_correction=True,
        )
    )

    rss_start = _rss_bytes()
    rss_boundaries = [rss_start] if rss_start is not None else []
    compile_start = time.perf_counter()
    state, warmup_metrics = run_chunk(state, data, priors)
    warmup_metrics.all_finite.block_until_ready()
    compile_and_first_chunk_seconds = time.perf_counter() - compile_start
    if not bool(np.asarray(warmup_metrics.all_finite).all()):
        raise RuntimeError("Nonfinite state during the compiling warmup chunk")

    post_compile_warmup_start = time.perf_counter()
    for _ in range(args.warmup_sweeps // args.chunk_size - 1):
        state, warmup_metrics = run_chunk(state, data, priors)
        warmup_metrics.all_finite.block_until_ready()
        if not bool(np.asarray(warmup_metrics.all_finite).all()):
            raise RuntimeError("Nonfinite state during warmup")
        rss = _rss_bytes()
        if rss is not None:
            rss_boundaries.append(rss)
    post_compile_warmup_seconds = time.perf_counter() - post_compile_warmup_start

    sample_chunks = []
    sample_chunk_seconds = []
    sample_wall_start = time.perf_counter()
    for _ in range(args.sample_sweeps // args.chunk_size):
        chunk_start = time.perf_counter()
        state, metrics = run_chunk(state, data, priors)
        metrics.all_finite.block_until_ready()
        sample_chunks.append(jax.device_get(metrics))
        sample_chunk_seconds.append(time.perf_counter() - chunk_start)
        rss = _rss_bytes()
        if rss is not None:
            rss_boundaries.append(rss)
    sample_wall_seconds = time.perf_counter() - sample_wall_start
    sample_seconds = float(np.sum(sample_chunk_seconds))

    traces = {
        field: np.concatenate(
            [np.asarray(getattr(chunk, field)) for chunk in sample_chunks], axis=0
        )
        for field in sample_chunks[0]._fields
    }
    all_finite = bool(traces["all_finite"].all())
    positive_scales = bool(
        np.all(traces["sigma_a"] > 0.0)
        and np.all(traces["tau_b"] > 0.0)
        and np.all(traces["sigma_campaign"] > 0.0)
        and np.all(traces["sigma_item"] > 0.0)
    )
    positive_omega = bool(np.all(traces["omega_min"] > 0.0))
    positive_cholesky = bool(np.all(traces["min_person_cholesky_diagonal"] > 0.0))
    stable = all_finite and positive_scales and positive_omega and positive_cholesky
    observations = len(panel["y"])
    device = jax.devices()[0]
    receipt = {
        "schema_version": "block-gibbs-benchmark-v1",
        "scientific_status": {
            "synthetic_data": True,
            "train_test_split": False,
            "structural_analog_not_paper_teacher": True,
            "substantive_interpretation_supported": False,
            "posterior_convergence_claimed": False,
            "purpose": "computational scalability and numerical stability",
        },
        "model_geometry": {
            "people": args.people,
            "observations": observations,
            "mean_observations_per_person": float(observations / args.people),
            "maximum_observations_per_person": int(panel["counts"].max()),
            "person_block_dimension": 4,
            "creative_coefficients_per_person": 3,
            "campaigns": args.campaigns,
            "items": args.items,
            "controls": args.controls,
            "respondent_features": args.respondent_features,
        },
        "sampler": {
            "algorithm": "systematic-scan conjugate block Gibbs",
            "pg_sampler": "fixed-series exponential with exact tail-mean correction",
            "pg_num_terms": args.num_terms,
            "dtype": args.dtype,
            "warmup_sweeps": args.warmup_sweeps,
            "sample_sweeps": args.sample_sweeps,
            "chunk_size": args.chunk_size,
            "person_state_storage": "final state only",
            "retained_trace": "scalar and small global diagnostics only",
        },
        "timing": {
            "compile_and_first_warmup_chunk_seconds": compile_and_first_chunk_seconds,
            "post_compile_warmup_seconds": post_compile_warmup_seconds,
            "sample_seconds": sample_seconds,
            "sample_wall_seconds_including_rss_probes": sample_wall_seconds,
            "sweeps_per_second": args.sample_sweeps / sample_seconds,
            "chunk_sweeps_per_second": _metric_summary(
                args.chunk_size / np.asarray(sample_chunk_seconds)
            ),
            "observation_pg_updates_per_second": (
                observations * args.sample_sweeps / sample_seconds
            ),
            "person_block_updates_per_second": (
                args.people * args.sample_sweeps / sample_seconds
            ),
        },
        "stability": {
            "passed": stable,
            "all_states_finite": all_finite,
            "all_scales_positive": positive_scales,
            "all_omega_positive": positive_omega,
            "all_person_cholesky_diagonals_positive": positive_cholesky,
            "log_likelihood_per_observation": _metric_summary(
                traces["log_likelihood_per_observation"]
            ),
            "omega_mean": _metric_summary(traces["omega_mean"]),
            "omega_min": _metric_summary(traces["omega_min"]),
            "omega_max": _metric_summary(traces["omega_max"]),
            "max_abs_eta": _metric_summary(traces["max_abs_eta"]),
            "min_person_cholesky_diagonal": _metric_summary(
                traces["min_person_cholesky_diagonal"]
            ),
            "scalar_trace_ess": {
                "mu_a": _effective_sample_size(traces["mu_a"]),
                "mu_b_0": _effective_sample_size(traces["mu_b"][:, 0]),
                "mu_b_1": _effective_sample_size(traces["mu_b"][:, 1]),
                "mu_b_2": _effective_sample_size(traces["mu_b"][:, 2]),
                "sigma_a": _effective_sample_size(traces["sigma_a"]),
                "sigma_campaign": _effective_sample_size(traces["sigma_campaign"]),
                "sigma_item": _effective_sample_size(traces["sigma_item"]),
            },
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": np.__version__,
            "psutil": importlib.metadata.version("psutil"),
            "package_version": __version__,
            "backend": jax.default_backend(),
            "device": str(device),
            "device_kind": getattr(device, "device_kind", None),
            "device_memory_stats_after_run": _json_safe_memory_stats(device),
            "process_rss_bytes_start": rss_start,
            "process_rss_bytes_max_at_chunk_boundaries": (
                max(rss_boundaries) if rss_boundaries else None
            ),
            "process_rss_bytes_end": _rss_bytes(),
        },
        "source_sha256": {
            "benchmarks/benchmark_block_gibbs.py": _sha256_file(Path(__file__)),
            "src/fast_polya_gamma_jax/block_gibbs.py": _sha256_file(
                PROJECT_ROOT / "src" / "fast_polya_gamma_jax" / "block_gibbs.py"
            ),
            "src/fast_polya_gamma_jax/sampler.py": _sha256_file(
                PROJECT_ROOT / "src" / "fast_polya_gamma_jax" / "sampler.py"
            ),
        },
        "seed": args.seed,
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not stable or not math.isfinite(receipt["timing"]["sweeps_per_second"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
