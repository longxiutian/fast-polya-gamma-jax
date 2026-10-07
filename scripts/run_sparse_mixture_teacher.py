"""Fit one registered NBE split with the PG sparse finite-mixture teacher."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import traceback
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import jax
import numpy as np

from fast_polya_gamma_jax.sparse_mixture_gibbs import (
    initialize_mixture_state,
    original_micro_priors,
    prepare_mixture_data,
    run_sparse_mixture,
    run_sparse_mixture_collect,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(json.dumps(values.shape).encode("ascii"))
    if values.dtype.hasobject:
        payload = json.dumps(
            values.tolist(), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    else:
        payload = values.tobytes(order="C")
    digest.update(payload)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def verified_split(prep: Path, index: int):
    master_path = prep / "MICRO_PANEL_PREPARATION_COMPLETED.json"
    master = json.loads(master_path.read_text())
    registry_path = prep / "registry.csv"
    if sha256_file(registry_path) != master["registry_sha256"]:
        raise ValueError("Preparation registry hash mismatch")
    with registry_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    record = next(
        (row for row in rows if int(row["configuration_index"]) == index), None
    )
    if record is None or len(rows) != 4:
        raise ValueError("Unknown four-split registry index")
    source = prep / record["configuration_id"]
    paths = {
        "configuration_manifest": source / "configuration_manifest.json",
        "micro_panel_manifest": source / "micro_panel_manifest.json",
    }
    expected = {
        "configuration_manifest": record["manifest_sha256"],
        "micro_panel_manifest": record["micro_panel_manifest_sha256"],
    }
    for name, path in paths.items():
        if sha256_file(path) != expected[name]:
            raise ValueError(f"{name} hash mismatch")
    configuration = json.loads(paths["configuration_manifest"].read_text())
    manifest = json.loads(paths["micro_panel_manifest"].read_text())
    for name, digest in manifest["artifacts"].items():
        if sha256_file(source / name) != digest:
            raise ValueError(f"Preparation artifact hash mismatch: {name}")
    with np.load(source / "micro_panel_pack.npz", allow_pickle=False) as archive:
        pack = {name: np.asarray(archive[name]) for name in archive.files}
    for name, digest in manifest["array_sha256"].items():
        if sha256_array(pack[name]) != digest:
            raise ValueError(f"Preparation array hash mismatch: {name}")
    levels = json.loads((source / "level_registry.json").read_text())
    from mengyao_nbe.refinement.micro_panel_teacher import validate_pack

    validate_pack(
        pack, len(levels["campaigns"]), len(levels["ads"]),
        minimum_training_observations=1,
    )
    counts = manifest["teacher_counts"]
    if (
        len(pack["A"]) != counts["sampled_users"]
        or int(pack["mask"].sum()) != counts["sampled_impressions"]
        or counts["person_subsampling"]
        or counts["minimum_training_exposures"] != 1
        or not manifest["teacher_sample_is_full_population"]
        or manifest["heldout_outcomes_in_fit_preparation"] is not False
    ):
        raise ValueError("Input is not registered full-person outcome-free training")
    return record, source, configuration, manifest, levels, pack, expected


def flat_training_arrays(pack: dict[str, np.ndarray]):
    mask = np.asarray(pack["mask"], dtype=bool)
    people, slots = mask.shape
    person_index = np.repeat(np.arange(people, dtype=np.int32), slots)[
        mask.ravel()
    ]
    return {
        "y": np.asarray(pack["y"][mask], dtype=np.float64),
        "person_index": person_index,
        "campaign_index": np.asarray(pack["campaign_index"][mask], dtype=np.int32),
        "item_index": np.asarray(pack["ad_index"][mask], dtype=np.int32),
        "creative_features": np.asarray(pack["Z"][mask], dtype=np.float64),
        "controls": np.asarray(pack["D"][mask], dtype=np.float64),
        "respondent_features": np.asarray(pack["A"], dtype=np.float64),
    }


def run(args: argparse.Namespace) -> dict:
    schedules = {
        "pilot": (20, 20, 10, 1, 10),
        "production": (2500, 20000, 100, 1, 10),
        "pilot_long": (20, 1000, 1000, 20, 200),
        "production_long": (20000, 1000000, 1000, 20, 200),
    }
    actual = (
        args.warmup, args.samples, args.chunk_size,
        args.global_archive_stride, args.archive_stride,
    )
    if actual != schedules[args.mode]:
        raise ValueError(f"{args.mode} schedule must equal {schedules[args.mode]}")
    if args.components != 8 or args.samples % args.chunk_size or (
        args.chunk_size % args.archive_stride
        or args.chunk_size % args.global_archive_stride
        or args.archive_stride % args.global_archive_stride
    ):
        raise ValueError("Mixture and archive contract mismatch")
    if not 0 <= args.index <= 3 or not 0 <= args.chain <= 3:
        raise ValueError("Split or chain index is outside contract")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    device = jax.devices()[0]
    if jax.default_backend() != "gpu" or not jax.config.x64_enabled:
        raise RuntimeError("Production requires float64 JAX GPU sampling")

    prep = args.prep_s12 if args.index < 2 else args.prep_s34
    record, source, configuration, manifest, levels, pack, expected = verified_split(
        prep, args.index
    )
    values = flat_training_arrays(pack)
    data = prepare_mixture_data(**values)
    people = len(pack["A"])
    if (
        len(values["y"]) != manifest["training_observations"]
        or len(values["y"]) != manifest["teacher_counts"]["sampled_impressions"]
    ):
        raise ValueError("Flattened training rows differ from source manifest")
    priors = original_micro_priors(
        covariates=pack["A"].shape[1],
        dirichlet_concentration=args.dirichlet_concentration,
    )
    seed = 2026092201 + args.index * 10 + args.chain
    state = initialize_mixture_state(
        jax.random.key(seed), people=people, campaigns=len(levels["campaigns"]),
        items=len(levels["ads"]), controls=pack["D"].shape[2],
        covariates=pack["A"].shape[1], components=args.components,
    )
    output = args.run_root / record["configuration_id"] / f"chain-{args.chain}"
    output.mkdir(parents=True, exist_ok=False)
    environment = {
        "backend": jax.default_backend(), "device": str(device),
        "device_kind": device.device_kind, "x64_enabled": bool(jax.config.x64_enabled),
        "jax_version": jax.__version__, "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "sampling_precision": "float64", "heldout_outcomes_read": False,
        "started_at": datetime.now(UTC).isoformat(),
    }
    atomic_json(output / "environment.json", environment)
    config = {
        "model": "pg_sparse_finite_gaussian_mixture_micro_panel_v1",
        "algorithm": "PG collapsed-label Metropolis-within-block-Gibbs",
        "mode": args.mode, "configuration": configuration["configuration"],
        "chain": args.chain, "seed": seed, "warmup": args.warmup,
        "samples": args.samples, "chunk_size": args.chunk_size,
        "global_archive_stride": args.global_archive_stride,
        "components": args.components,
        "dirichlet_concentration": args.dirichlet_concentration,
        "component_sd": float(priors.component_sd),
        "archive_stride": args.archive_stride,
        "pg_num_terms": args.num_terms,
        "pg_draws_approximate": True,
        "original_non_mixture_priors_retained": True,
        "fresh_chain": True, "heldout_outcomes_read": False,
        "source_prep_root": str(prep), "source_split_root": str(source),
        "source_manifest_sha256": expected,
        "source_pack_sha256": manifest["artifacts"]["micro_panel_pack.npz"],
        "source_people": people, "source_observations": len(values["y"]),
        "source_audience_columns": pack["A"].shape[1],
        "source_control_columns": pack["D"].shape[2],
        "source_campaigns": len(levels["campaigns"]),
        "source_ads": len(levels["ads"]),
    }
    atomic_json(output / "config.json", config)
    del pack, values

    warmup_chunk = jax.jit(partial(
        run_sparse_mixture, num_steps=args.chunk_size, num_terms=args.num_terms
    ))
    collector = jax.jit(partial(
        run_sparse_mixture_collect, num_steps=args.chunk_size,
        archive_stride=args.archive_stride, num_terms=args.num_terms,
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
        state, metrics, globals_, summed, squared, archive = collector(
            state, data, priors
        )
        metrics.all_finite.block_until_ready()
        metric_host = jax.device_get(metrics)
        if not bool(np.asarray(metric_host.all_finite).all()):
            raise RuntimeError(f"Nonfinite sample chunk {number}")
        global_host = jax.device_get(globals_)
        archive_host = np.asarray(jax.device_get(archive))
        person_sum += np.asarray(jax.device_get(summed))
        person_square_sum += np.asarray(jax.device_get(squared))
        first = number * args.chunk_size
        last = first + args.chunk_size
        global_draw_index = np.arange(
            first + args.global_archive_stride - 1, last,
            args.global_archive_stride, dtype=np.int64,
        )
        draw_index = np.arange(
            first + args.archive_stride - 1, last, args.archive_stride,
            dtype=np.int64,
        )
        if archive_host.shape != (len(draw_index), people, 4):
            raise RuntimeError("Respondent archive is misaligned")
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
            "global_draw_index": global_draw_index,
            "person_draw_index": draw_index,
            "person_archive": archive_host,
        }
        chunk_path = output / f"chunk-{number:03d}.npz"
        atomic_npz(chunk_path, payload)
        entry = {
            "number": number, "draw_start": first, "draw_stop": last,
            "posterior_draws": args.chunk_size,
            "global_archive_draws": len(global_draw_index),
            "person_archive_draws": len(draw_index),
            "sha256": sha256_file(chunk_path),
            "bytes": chunk_path.stat().st_size,
            "elapsed_seconds": time.perf_counter() - t_chunk,
            "component_occupancy_min": int(
                np.min(metric_host.occupied_components)
            ),
            "component_occupancy_max": int(
                np.max(metric_host.occupied_components)
            ),
        }
        atomic_json(output / f"chunk-{number:03d}.json", entry)
        chunks.append(entry)
        print(
            f"CHUNK {number} draws={last} seconds={entry['elapsed_seconds']:.2f} "
            f"bytes={entry['bytes']}", flush=True,
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
        "schema_version": "pg-sparse-mixture-chain-v1",
        "model": config["model"], "mode": args.mode,
        "configuration_id": record["configuration_id"],
        "chain": args.chain, "seed": seed,
        "warmup_completed": args.warmup,
        "retained_draws": args.samples,
        "archived_global_draws": args.samples // args.global_archive_stride,
        "archived_person_draws": args.samples // args.archive_stride,
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
        "configuration_id": record["configuration_id"], "chain": args.chain,
        "chain_manifest_sha256": sha256_file(manifest_path),
        "retained_draws": args.samples,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    atomic_json(output / "MIXTURE_CHAIN_COMPLETED.json", marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-s12", type=Path, required=True)
    parser.add_argument("--prep-s34", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--chain", type=int, default=0)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("pilot", "production", "pilot_long", "production_long"),
        required=True,
    )
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--global-archive-stride", type=int, default=1)
    parser.add_argument("--archive-stride", type=int, default=10)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--dirichlet-concentration", type=float, default=0.01)
    parser.add_argument("--num-terms", type=int, default=8)
    args = parser.parse_args()
    try:
        marker = run(args)
    except Exception:
        failure = {
            "status": "failed", "index": args.index, "chain": args.chain,
            "at": datetime.now(UTC).isoformat(),
            "traceback": traceback.format_exc(),
        }
        failure_dir = args.run_root / "_failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(failure_dir / f"task-{args.index}-chain-{args.chain}.json", failure)
        raise
    print(json.dumps(marker, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
