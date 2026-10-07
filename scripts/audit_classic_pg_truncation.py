"""Audit the omitted PG-series variance at verified classical-HB logits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CONFIGURATIONS = (
    "global-exposure80-seed2026090902",
    "within-campaign-last20pct-weeks-v1",
    "within-campaign-one-ad-seed2026091103",
    "hybrid-one-ad-plus-known-exposure20-seed2026091504",
)
CHUNKS = (99, 199, 299, 399, 499, 599, 699, 799, 899, 999)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tail_variance_fraction(abs_eta: np.ndarray, terms: int) -> np.ndarray:
    """Fraction of exact PG(1, eta) variance omitted by the finite series."""
    magnitude = np.asarray(abs_eta, dtype=np.float64)
    safe = np.where(magnitude < 0.01, 1.0, magnitude)
    half_tanh = np.tanh(safe / 2.0)
    exact = (
        half_tanh - 0.5 * safe * (1.0 - half_tanh**2)
    ) / (2.0 * safe**3)
    exact = np.where(
        magnitude < 0.01,
        1.0 / 24.0 - magnitude**2 / 120.0,
        exact,
    )
    tilt = (magnitude[:, None] / (2.0 * np.pi)) ** 2
    half_integer = np.arange(terms, dtype=np.float64) + 0.5
    denominator = half_integer[None, :] ** 2 + tilt
    retained = np.sum(denominator**-2, axis=1) / (4.0 * np.pi**4)
    return np.clip((exact - retained) / exact, 0.0, 1.0)


def training_design(config: dict) -> tuple[np.ndarray, ...]:
    pack_path = Path(config["source_split_root"]) / "micro_panel_pack.npz"
    if digest(pack_path) != config["source_pack_sha256"]:
        raise ValueError("Frozen training pack hash mismatch")
    with np.load(pack_path, allow_pickle=False) as archive:
        mask = np.asarray(archive["mask"], dtype=bool)
        people, slots = mask.shape
        person = np.repeat(np.arange(people, dtype=np.int32), slots)[mask.ravel()]
        creative = np.asarray(archive["Z"][mask], dtype=np.float64)
        controls = np.asarray(archive["D"][mask], dtype=np.float64)
        campaign = np.asarray(archive["campaign_index"][mask], dtype=np.int32)
        ad = np.asarray(archive["ad_index"][mask], dtype=np.int32)
    if people != config["source_people"] or len(person) != config["source_observations"]:
        raise ValueError("Training pack dimensions differ from fitted chain")
    return person, creative, controls, campaign, ad


def chain_logits(root: Path, config_id: str) -> tuple[np.ndarray, dict]:
    config = json.loads((root / "config.json").read_text())
    manifest_path = root / "chain_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    marker = json.loads((root / "CLASSIC_CHAIN_VERIFIED.json").read_text())
    if (
        marker["status"] != "verified"
        or marker["chain_manifest_sha256"] != digest(manifest_path)
        or config["configuration"]["configuration_id"] != config_id
        or config["mode"] != "production"
        or config["samples"] != 1_000_000
        or config["chunk_size"] != 1_000
        or config["person_archive_stride"] != 1_000
        or config["pg_num_terms"] != 8
        or len(manifest["chunks"]) != 1_000
    ):
        raise ValueError(f"PG chain does not meet the audit contract: {root}")
    person_index, creative, controls, campaign_index, ad_index = training_design(config)
    logits = []
    chunk_hashes = []
    for number in CHUNKS:
        entry = manifest["chunks"][number]
        path = root / f"chunk-{number:03d}.npz"
        chunk_hash = digest(path)
        if entry["sha256"] != chunk_hash or entry["number"] != number:
            raise ValueError(f"Archived draw hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            expected_draw = (number + 1) * 1_000 - 1
            if (
                not np.array_equal(archive["person_draw_index"], [expected_draw])
                or archive["global_draw_index"][-1] != expected_draw
            ):
                raise ValueError(f"Person and global draws are not aligned: {path}")
            person = np.asarray(archive["person_archive"][0], dtype=np.float64)
            eta = (
                person[person_index, 0]
                + np.sum(person[person_index, 1:] * creative, axis=1)
                + np.asarray(archive["campaign_effect"][-1])[campaign_index]
                + np.asarray(archive["ad_effect"][-1])[ad_index]
                + controls @ np.asarray(archive["delta"][-1])
            )
        if not np.isfinite(eta).all():
            raise ValueError(f"Nonfinite training logits in {path}")
        logits.append(np.abs(eta))
        chunk_hashes.append(chunk_hash)
    return np.concatenate(logits), {
        "chain_manifest_sha256": digest(manifest_path),
        "training_pack_sha256": config["source_pack_sha256"],
        "audited_chunk_sha256": chunk_hashes,
    }


def run(run_root: Path, index: int, output_root: Path) -> dict:
    config_id = CONFIGURATIONS[index]
    arrays = []
    sources = []
    for chain in (0, 1):
        values, source = chain_logits(run_root / config_id / f"chain-{chain}", config_id)
        arrays.append(values)
        sources.append(source)
    magnitude = np.concatenate(arrays)
    fraction = tail_variance_fraction(magnitude, 8)
    if not np.isfinite(fraction).all():
        raise ValueError("Nonfinite PG tail-variance diagnostic")
    result = {
        "schema_version": "classic-hb-pg-truncation-audit-v1",
        "configuration_id": config_id,
        "pg_terms": 8,
        "audited_draws_per_chain": len(CHUNKS),
        "training_logits_evaluated": len(magnitude),
        "absolute_logit_quantiles": {
            str(q): float(np.quantile(magnitude, q))
            for q in (0.5, 0.9, 0.99, 0.999, 1.0)
        },
        "fraction_abs_logit_above_10": float(np.mean(magnitude > 10)),
        "fraction_abs_logit_above_20": float(np.mean(magnitude > 20)),
        "omitted_pg_variance_fraction_quantiles": {
            str(q): float(np.quantile(fraction, q))
            for q in (0.5, 0.9, 0.99, 0.999, 1.0)
        },
        "mean_omitted_pg_variance_fraction": float(np.mean(fraction)),
        "interpretation": (
            "Latent-PG variance diagnostic only; this is not a bound on "
            "posterior or predictive bias from the approximate PG draw."
        ),
        "sources": sources,
    }
    output = output_root / config_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "pg_truncation_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "configuration_id": config_id,
        "fraction_abs_logit_above_10": result["fraction_abs_logit_above_10"],
        "mean_omitted_pg_variance_fraction": result[
            "mean_omitted_pg_variance_fraction"
        ],
    }, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--index", type=int, choices=range(4), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.run_root, args.index, args.output_root)


if __name__ == "__main__":
    main()
