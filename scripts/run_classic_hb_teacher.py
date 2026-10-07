"""Fit one registered split with the PG-augmented non-mixture HB teacher."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import jax
import numpy as np
from mengyao_nbe.refinement import micro_panel_teacher
from run_sparse_mixture_teacher import (
    atomic_json,
    atomic_npz,
    flat_training_arrays,
    sha256_file,
    verified_split,
)

from fast_polya_gamma_jax.classic_hb_gibbs import (
    initialize_classic_state,
    original_classic_priors,
    prepare_classic_data,
    run_classic,
    run_classic_collect,
)

SCHEDULES = {
    "smoke": (20, 20, 10, 1, 10),
    "pilot": (100, 1000, 100, 10, 100),
    "production": (20000, 1000000, 1000, 20, 1000),
}


def verify_prior_contract(priors, covariates: int) -> None:
    original = micro_panel_teacher.PRIORS
    expected = {
        "mu_a_sd": float(priors.mu_a_sd),
        "sigma_a_halfnormal_scale": float(priors.sigma_a_scale),
        "mu_B_sd": float(priors.mu_b_sd),
        "tau_B_halfstudent_df": float(priors.sigma_b_df),
        "tau_B_halfstudent_scale": float(priors.sigma_b_scale),
        "lkj_concentration": float(priors.lkj_concentration),
        "delta_sd": float(priors.delta_sd),
        "campaign_halfnormal_scale": float(priors.sigma_campaign_scale),
        "ad_halfnormal_scale": float(priors.sigma_item_scale),
        "Gamma_slab_sd": float(priors.gamma_slab_sd),
    }
    if any(not np.isclose(original[name], value) for name, value in expected.items()):
        raise ValueError("Classical HB prior constants differ from HMC")
    if (
        original["Gamma_global_halfcauchy_scale_rule"]
        != "0.1/sqrt(number_of_frozen_respondent_columns)"
        or not np.isclose(
            float(priors.gamma_global_scale), 0.1 / np.sqrt(covariates)
        )
        or original["Gamma_local_halfcauchy_scale"] != 1.0
    ):
        raise ValueError("Classical HB horseshoe prior differs from HMC")


def code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    source = root / "src/fast_polya_gamma_jax"
    return {
        "runner": sha256_file(Path(__file__).resolve()),
        "classic_kernel": sha256_file(source / "classic_hb_gibbs.py"),
        "pg_sampler": sha256_file(source / "sampler.py"),
        "mixture_helpers": sha256_file(source / "sparse_mixture_gibbs.py"),
        "block_helpers": sha256_file(source / "block_gibbs.py"),
        "hmc_model": sha256_file(Path(micro_panel_teacher.__file__)),
    }


def run(args: argparse.Namespace) -> dict:
    actual = (
        args.warmup, args.samples, args.chunk_size,
        args.global_archive_stride, args.person_archive_stride,
    )
    if actual != SCHEDULES[args.mode]:
        raise ValueError(f"{args.mode} schedule must equal {SCHEDULES[args.mode]}")
    if (
        not 0 <= args.index <= 3 or not 0 <= args.chain <= 3
        or args.num_terms not in (8, 16, 32)
    ):
        raise ValueError("Split, chain, or PG truncation outside the contract")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    device = jax.devices()[0]
    if jax.default_backend() != "gpu" or not jax.config.x64_enabled:
        raise RuntimeError("The registered run requires float64 JAX on GPU")

    prep = args.prep_s12 if args.index < 2 else args.prep_s34
    record, source, configuration, manifest, levels, pack, expected = verified_split(
        prep, args.index
    )
    values = flat_training_arrays(pack)
    data = prepare_classic_data(**values)
    people = len(pack["A"])
    if len(values["y"]) != manifest["training_observations"]:
        raise ValueError("Training row count differs from frozen manifest")
    covariates = pack["A"].shape[1]
    controls = pack["D"].shape[2]
    priors = original_classic_priors(covariates=covariates)
    verify_prior_contract(priors, covariates)
    seed = 2026092301 + args.index * 10 + args.chain
    state = initialize_classic_state(
        jax.random.key(seed), people=people,
        campaigns=len(levels["campaigns"]), items=len(levels["ads"]),
        controls=controls, covariates=covariates,
    )
    output = args.run_root / record["configuration_id"] / f"chain-{args.chain}"
    output.mkdir(parents=True, exist_ok=False)
    environment = {
        "backend": jax.default_backend(), "device": str(device),
        "device_kind": device.device_kind,
        "x64_enabled": bool(jax.config.x64_enabled),
        "jax_version": jax.__version__,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "sampling_precision": "float64", "heldout_outcomes_read": False,
        "started_at": datetime.now(UTC).isoformat(),
    }
    atomic_json(output / "environment.json", environment)
    config = {
        "model": "pg_classic_micro_panel_hb_v1",
        "algorithm": "PG block Gibbs with MH hyperparameters and location/scale ASIS",
        "algorithm_version": "offcentered-locations-scales-v2",
        "pg_draws_approximate": True,
        "pg_num_terms": args.num_terms,
        "mode": args.mode, "configuration": configuration["configuration"],
        "chain": args.chain, "seed": seed,
        "warmup": args.warmup, "samples": args.samples,
        "chunk_size": args.chunk_size,
        "global_archive_stride": args.global_archive_stride,
        "person_archive_stride": args.person_archive_stride,
        "off_centered_location_interweaving": True,
        "off_centered_scale_interweaving": True,
        "scale_proposal_log_sd": 0.01,
        "original_hmc_prior_contract": True,
        "fresh_chain": True, "heldout_outcomes_read": False,
        "source_prep_root": str(prep), "source_split_root": str(source),
        "source_manifest_sha256": expected,
        "source_pack_sha256": manifest["artifacts"]["micro_panel_pack.npz"],
        "source_people": people, "source_observations": len(values["y"]),
        "source_audience_columns": covariates,
        "source_control_columns": controls,
        "source_campaigns": len(levels["campaigns"]),
        "source_ads": len(levels["ads"]),
        "code_sha256": code_hashes(),
    }
    atomic_json(output / "config.json", config)
    del pack, values

    warmup_chunk = jax.jit(partial(
        run_classic, num_steps=args.chunk_size, num_terms=args.num_terms,
    ))
    collector = jax.jit(partial(
        run_classic_collect, num_steps=args.chunk_size,
        archive_stride=args.person_archive_stride,
        num_terms=args.num_terms,
    ))
    t0 = time.perf_counter()
    for number in range(args.warmup // args.chunk_size):
        state, metrics = warmup_chunk(state, data, priors)
        metrics.all_finite.block_until_ready()
        if not bool(np.asarray(metrics.all_finite).all()):
            raise RuntimeError(f"Nonfinite warmup chunk {number}")
        print(f"WARMUP {number + 1}/{args.warmup // args.chunk_size}", flush=True)
    warmup_seconds = time.perf_counter() - t0

    person_sum = np.zeros((people, 4), dtype=np.float64)
    person_square_sum = np.zeros((people, 4), dtype=np.float64)
    chunks = []
    sampling_start = time.perf_counter()
    for number in range(args.samples // args.chunk_size):
        t_chunk = time.perf_counter()
        state, metrics, global_trace, summed, squared, archive = collector(
            state, data, priors
        )
        metrics.all_finite.block_until_ready()
        metric_host = jax.device_get(metrics)
        if not bool(np.asarray(metric_host.all_finite).all()):
            raise RuntimeError(f"Nonfinite sample chunk {number}")
        global_host = jax.device_get(global_trace)
        archive_host = np.asarray(jax.device_get(archive))
        person_sum += np.asarray(jax.device_get(summed))
        person_square_sum += np.asarray(jax.device_get(squared))
        first = number * args.chunk_size
        last = first + args.chunk_size
        global_index = np.arange(
            first + args.global_archive_stride - 1, last,
            args.global_archive_stride, dtype=np.int64,
        )
        person_index = np.arange(
            first + args.person_archive_stride - 1, last,
            args.person_archive_stride, dtype=np.int64,
        )
        if archive_host.shape != (len(person_index), people, 4):
            raise RuntimeError("Respondent archive shape mismatch")
        payload = {
            **{
                name: np.asarray(value)[args.global_archive_stride - 1::
                                        args.global_archive_stride]
                for name, value in global_host.items()
            },
            **{
                "metric_" + name: np.asarray(getattr(metric_host, name))[
                    args.global_archive_stride - 1::args.global_archive_stride
                ]
                for name in metric_host._fields
            },
            "global_draw_index": global_index,
            "person_draw_index": person_index,
            "person_archive": archive_host,
        }
        chunk_path = output / f"chunk-{number:03d}.npz"
        atomic_npz(chunk_path, payload)
        entry = {
            "number": number, "draw_start": first, "draw_stop": last,
            "posterior_draws": args.chunk_size,
            "global_archive_draws": len(global_index),
            "person_archive_draws": len(person_index),
            "sha256": sha256_file(chunk_path),
            "bytes": chunk_path.stat().st_size,
            "elapsed_seconds": time.perf_counter() - t_chunk,
        }
        atomic_json(output / f"chunk-{number:03d}.json", entry)
        chunks.append(entry)
        print(
            f"CHUNK {number} draws={last} seconds={entry['elapsed_seconds']:.2f}",
            flush=True,
        )
    sampling_seconds = time.perf_counter() - sampling_start
    mean = person_sum / args.samples
    variance = (
        person_square_sum - args.samples * np.square(mean)
    ) / (args.samples - 1)
    moments_path = output / "person_posterior_moments.npz"
    atomic_npz(moments_path, {
        "a_mean": mean[:, 0],
        "a_sd": np.sqrt(np.maximum(variance[:, 0], 0.0)),
        "B_mean": mean[:, 1:],
        "B_sd": np.sqrt(np.maximum(variance[:, 1:], 0.0)),
    })
    manifest_payload = {
        "schema_version": "pg-classic-hb-chain-v1",
        "model": config["model"], "mode": args.mode,
        "configuration_id": record["configuration_id"],
        "chain": args.chain, "seed": seed,
        "warmup_completed": args.warmup,
        "retained_draws": args.samples,
        "archived_global_draws": args.samples // args.global_archive_stride,
        "archived_person_draws": args.samples // args.person_archive_stride,
        "all_draws_used_for_person_moments": True,
        "all_draws_used_for_predictions": False,
        "heldout_outcomes_read": False,
        "config_sha256": sha256_file(output / "config.json"),
        "environment_sha256": sha256_file(output / "environment.json"),
        "person_moments_sha256": sha256_file(moments_path),
        "chunks": chunks,
        "timing": {
            "warmup_seconds": warmup_seconds,
            "sampling_seconds": sampling_seconds,
        },
    }
    manifest_path = output / "chain_manifest.json"
    atomic_json(manifest_path, manifest_payload)
    marker = {
        "status": "complete", "model": config["model"],
        "configuration_id": record["configuration_id"],
        "chain": args.chain,
        "chain_manifest_sha256": sha256_file(manifest_path),
        "retained_draws": args.samples,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    atomic_json(output / "CLASSIC_CHAIN_COMPLETED.json", marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-s12", type=Path, required=True)
    parser.add_argument("--prep-s34", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--chain", type=int, default=0)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=SCHEDULES, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--global-archive-stride", type=int, required=True)
    parser.add_argument("--person-archive-stride", type=int, required=True)
    parser.add_argument("--num-terms", type=int, default=8)
    args = parser.parse_args()
    try:
        marker = run(args)
    except Exception:
        failure_dir = args.run_root / "_failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(failure_dir / f"task-{args.index}-chain-{args.chain}.json", {
            "status": "failed", "index": args.index, "chain": args.chain,
            "at": datetime.now(UTC).isoformat(),
            "traceback": traceback.format_exc(),
        })
        raise
    print(json.dumps(marker, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
