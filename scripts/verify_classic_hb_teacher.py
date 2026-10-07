"""Verify one PG classical-HB chain and its registered artifact hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from run_classic_hb_teacher import SCHEDULES, code_hashes
from run_sparse_mixture_teacher import atomic_json, sha256_file


def verify_chain(root: Path) -> dict:
    required = (
        "config.json", "environment.json", "chain_manifest.json",
        "CLASSIC_CHAIN_COMPLETED.json", "person_posterior_moments.npz",
    )
    if any(not (root / name).is_file() for name in required):
        raise ValueError("Chain is missing a required artifact")
    config = json.loads((root / "config.json").read_text())
    environment = json.loads((root / "environment.json").read_text())
    manifest = json.loads((root / "chain_manifest.json").read_text())
    completion = json.loads((root / "CLASSIC_CHAIN_COMPLETED.json").read_text())
    if (
        config["model"] != "pg_classic_micro_panel_hb_v1"
        or not config["original_hmc_prior_contract"]
        or not config["pg_draws_approximate"]
        or not config["off_centered_location_interweaving"]
        or not config["off_centered_scale_interweaving"]
        or config["algorithm_version"] != "offcentered-locations-scales-v2"
        or config["heldout_outcomes_read"] is not False
        or environment["backend"] != "gpu"
        or environment["sampling_precision"] != "float64"
        or not environment["x64_enabled"]
        or environment["heldout_outcomes_read"] is not False
        or manifest["heldout_outcomes_read"] is not False
        or not manifest["all_draws_used_for_person_moments"]
        or manifest["all_draws_used_for_predictions"] is not False
        or config["code_sha256"] != code_hashes()
    ):
        raise ValueError("Model, environment, or data-access contract mismatch")
    actual_schedule = (
        config["warmup"], config["samples"], config["chunk_size"],
        config["global_archive_stride"], config["person_archive_stride"],
    )
    if actual_schedule != SCHEDULES[config["mode"]]:
        raise ValueError("Sampling schedule mismatch")
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
        raise ValueError("Completion hashes do not bind the chain artifacts")
    people = config["source_people"]
    draws = config["samples"]
    chunk_size = config["chunk_size"]
    global_stride = config["global_archive_stride"]
    person_stride = config["person_archive_stride"]
    chunks = manifest["chunks"]
    if len(chunks) != draws // chunk_size:
        raise ValueError("Chunk count mismatch")
    seen_global = 0
    seen_person = 0
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
            or entry["draw_start"] != number * chunk_size
            or entry["draw_stop"] != (number + 1) * chunk_size
        ):
            raise ValueError(f"Chunk {number} receipt mismatch")
        first = number * chunk_size
        expected_global = np.arange(
            first + global_stride - 1, first + chunk_size, global_stride,
            dtype=np.int64,
        )
        expected_person = np.arange(
            first + person_stride - 1, first + chunk_size, person_stride,
            dtype=np.int64,
        )
        with np.load(path, allow_pickle=False) as archive:
            if (
                not np.array_equal(archive["global_draw_index"], expected_global)
                or not np.array_equal(archive["person_draw_index"], expected_person)
                or archive["person_archive"].shape != (
                    len(expected_person), people, 4
                )
                or archive["person_archive"].dtype != np.float32
                or archive["Gamma"].shape != (
                    len(expected_global), 3, config["source_audience_columns"]
                )
                or archive["Gamma_global"].shape != (len(expected_global), 3)
                or archive["Gamma_local"].shape != (
                    len(expected_global), 3, config["source_audience_columns"]
                )
                or archive["Sigma_B"].shape != (len(expected_global), 3, 3)
                or archive["mu_a"].dtype != np.float64
                or not archive["metric_all_finite"].all()
                or not (archive["metric_omega_min"] > 0).all()
                or not (archive["metric_person_cholesky_min"] > 0).all()
                or archive["metric_off_centered_scale_acceptance"].shape != (
                    len(expected_global), 4
                )
            ):
                raise ValueError(f"Chunk {number} numerical or shape mismatch")
            if any(not np.isfinite(archive[name]).all() for name in archive.files):
                raise ValueError(f"Chunk {number} contains nonfinite values")
        seen_global += len(expected_global)
        seen_person += len(expected_person)
    if (
        seen_global != draws // global_stride
        or seen_person != draws // person_stride
        or manifest["retained_draws"] != draws
        or manifest["archived_global_draws"] != seen_global
        or manifest["archived_person_draws"] != seen_person
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
        "schema_version": "pg-classic-hb-verification-v1",
        "status": "verified", "mode": config["mode"],
        "configuration_id": config["configuration"]["configuration_id"],
        "chain": config["chain"], "people": people,
        "retained_draws": draws,
        "archived_global_draws": seen_global,
        "archived_person_draws": seen_person,
        "chunks_verified": len(chunks),
        "chain_manifest_sha256": sha256_file(root / "chain_manifest.json"),
        "heldout_outcomes_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-root", type=Path, required=True)
    parser.add_argument("--write-marker", action="store_true")
    args = parser.parse_args()
    receipt = verify_chain(args.chain_root)
    if args.write_marker:
        path = args.chain_root / "CLASSIC_CHAIN_VERIFIED.json"
        if path.exists():
            raise FileExistsError(path)
        atomic_json(path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
