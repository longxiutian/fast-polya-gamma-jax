"""Independently verify one PG sparse-mixture teacher chain and its hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from run_sparse_mixture_teacher import atomic_json, sha256_file


def verify_chain(root: Path) -> dict:
    required = (
        "config.json", "environment.json", "chain_manifest.json",
        "MIXTURE_CHAIN_COMPLETED.json", "person_posterior_moments.npz",
    )
    if any(not (root / name).is_file() for name in required):
        raise ValueError("Chain is missing a required artifact")
    if (root / "MIXTURE_CHAIN_FAILED.json").exists():
        raise ValueError("Chain has a failure marker")
    config = json.loads((root / "config.json").read_text())
    environment = json.loads((root / "environment.json").read_text())
    manifest = json.loads((root / "chain_manifest.json").read_text())
    completion = json.loads((root / "MIXTURE_CHAIN_COMPLETED.json").read_text())
    if (
        environment["backend"] != "gpu"
        or not environment["device"].startswith("cuda:")
        or environment["sampling_precision"] != "float64"
        or not environment["x64_enabled"]
        or environment["heldout_outcomes_read"] is not False
        or config["heldout_outcomes_read"] is not False
        or not config["fresh_chain"]
        or not config["original_non_mixture_priors_retained"]
        or not config["pg_draws_approximate"]
        or manifest["heldout_outcomes_read"] is not False
        or not manifest["all_draws_used_for_person_moments"]
    ):
        raise ValueError("Sampling environment or model attestation mismatch")
    schedules = {
        "pilot": (20, 20, 10, 1, 10),
        "production": (2500, 20000, 100, 1, 10),
        "pilot_long": (20, 1000, 1000, 20, 200),
        "production_long": (20000, 1000000, 1000, 20, 200),
    }
    if config["mode"] not in schedules:
        raise ValueError("Unknown chain mode")
    actual = (
        config["warmup"], config["samples"], config["chunk_size"],
        config.get("global_archive_stride", 1), config["archive_stride"],
    )
    if actual != schedules[config["mode"]]:
        raise ValueError("Sampling and retention schedule mismatch")
    if (
        completion["status"] != "complete"
        or completion["chain_manifest_sha256"] != sha256_file(
            root / "chain_manifest.json"
        )
        or manifest["config_sha256"] != sha256_file(root / "config.json")
        or manifest["environment_sha256"] != sha256_file(root / "environment.json")
        or manifest["person_moments_sha256"] != sha256_file(
            root / "person_posterior_moments.npz"
        )
    ):
        raise ValueError("Completion binding hash mismatch")
    people = config["source_people"]
    draws = config["samples"]
    stride = config["archive_stride"]
    global_stride = config.get("global_archive_stride", 1)
    chunks = manifest["chunks"]
    if len(chunks) != draws // config["chunk_size"]:
        raise ValueError("Chunk count differs from registered draw count")
    seen_draws = 0
    seen_global_archive = 0
    seen_person_archive = 0
    for number, entry in enumerate(chunks):
        path = root / f"chunk-{number:03d}.npz"
        receipt_path = root / f"chunk-{number:03d}.json"
        if not path.is_file() or not receipt_path.is_file():
            raise ValueError(f"Missing chunk {number}")
        if (
            json.loads(receipt_path.read_text()) != entry
            or sha256_file(path) != entry["sha256"]
            or path.stat().st_size != entry["bytes"]
            or entry["number"] != number
            or entry["draw_start"] != seen_draws
            or entry["draw_stop"] != seen_draws + config["chunk_size"]
        ):
            raise ValueError(f"Chunk {number} binding mismatch")
        with np.load(path, allow_pickle=False) as archive:
            index = archive["person_draw_index"]
            global_index = (
                archive["global_draw_index"]
                if "global_draw_index" in archive.files
                else np.arange(seen_draws, seen_draws + config["chunk_size"])
            )
            person = archive["person_archive"]
            expected_global_index = np.arange(
                seen_draws + global_stride - 1,
                seen_draws + config["chunk_size"], global_stride,
                dtype=np.int64,
            )
            expected_index = np.arange(
                seen_draws + stride - 1,
                seen_draws + config["chunk_size"],
                stride, dtype=np.int64,
            )
            if (
                not np.array_equal(global_index, expected_global_index)
                or not np.isin(index, global_index).all()
                or not np.array_equal(index, expected_index)
                or person.shape != (len(index), people, 4)
                or person.dtype != np.float32
                or not np.isfinite(person).all()
                or archive["Gamma"].shape != (
                    len(global_index), 3, config["source_audience_columns"]
                )
                or archive["component_mean"].shape != (
                    len(global_index), config["components"], 3
                )
                or archive["weights"].shape != (
                    len(global_index), config["components"]
                )
                or archive["mu_a"].dtype != np.float64
                or not bool(archive["metric_all_finite"].all())
                or not (archive["metric_omega_min"] > 0).all()
                or not (archive["metric_person_cholesky_min"] > 0).all()
            ):
                raise ValueError(f"Chunk {number} numerical or shape mismatch")
            if not np.allclose(archive["weights"].sum(axis=1), 1.0):
                raise ValueError(f"Chunk {number} mixture weights are invalid")
        seen_draws += config["chunk_size"]
        seen_global_archive += len(global_index)
        seen_person_archive += len(index)
    if (
        seen_draws != draws
        or seen_global_archive != draws // global_stride
        or seen_person_archive != draws // stride
        or manifest["retained_draws"] != draws
        or manifest.get("archived_global_draws", draws) != draws // global_stride
        or manifest["archived_person_draws"] != draws // stride
    ):
        raise ValueError("Posterior draw accounting mismatch")
    with np.load(root / "person_posterior_moments.npz", allow_pickle=False) as moments:
        if (
            moments["a_mean"].shape != (people,)
            or moments["a_sd"].shape != (people,)
            or moments["B_mean"].shape != (people, 3)
            or moments["B_sd"].shape != (people, 3)
            or any(not np.isfinite(moments[name]).all() for name in moments.files)
            or (moments["a_sd"] < 0).any()
            or (moments["B_sd"] < 0).any()
        ):
            raise ValueError("Person posterior moments are malformed")
    return {
        "schema_version": "pg-sparse-mixture-verification-v1",
        "status": "verified", "mode": config["mode"],
        "configuration_id": config["configuration"]["configuration_id"],
        "chain": config["chain"], "people": people,
        "retained_draws": seen_draws,
        "archived_global_draws": seen_global_archive,
        "archived_person_draws": seen_person_archive,
        "chunks_verified": len(chunks),
        "chain_manifest_sha256": sha256_file(root / "chain_manifest.json"),
        "completion_marker_sha256": sha256_file(
            root / "MIXTURE_CHAIN_COMPLETED.json"
        ),
        "all_draws_used_for_person_moments": True,
        "all_draws_used_for_predictions": False,
        "heldout_outcomes_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-root", type=Path, required=True)
    parser.add_argument("--write-marker", action="store_true")
    args = parser.parse_args()
    receipt = verify_chain(args.chain_root)
    if args.write_marker:
        path = args.chain_root / "MIXTURE_CHAIN_VERIFIED.json"
        if path.exists():
            raise FileExistsError(path)
        atomic_json(path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
