"""CPU-only, label-invariant diagnostics for two verified million-draw chains."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpyro.diagnostics import effective_sample_size, gelman_rubin
from scipy.special import ndtri
from scipy.stats import rankdata

CONFIGURATIONS = (
    "global-exposure80-seed2026090902",
    "within-campaign-last20pct-weeks-v1",
    "within-campaign-one-ad-seed2026091103",
    "hybrid-one-ad-plus-known-exposure20-seed2026091504",
)
PERSON_NAMES = ("a_i", "beta_brand", "beta_performance", "beta_duration30")
SAMPLES = 1_000_000
WARMUP = 20_000
CHUNK_SIZE = 1_000
CHUNKS = SAMPLES // CHUNK_SIZE
GLOBAL_STRIDE = 20
PERSON_STRIDE = 200
PERSON_ARCHIVE_PER_CHUNK = CHUNK_SIZE // PERSON_STRIDE
PERSON_ARCHIVED = SAMPLES // PERSON_STRIDE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(values: np.ndarray) -> dict:
    valid = np.asarray(values, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if not len(valid):
        return {"n": 0}
    return {
        "n": int(len(valid)),
        "min": float(valid.min()),
        "p05": float(np.quantile(valid, 0.05)),
        "median": float(np.median(valid)),
        "p95": float(np.quantile(valid, 0.95)),
        "max": float(valid.max()),
    }


def split_chains(draws: np.ndarray) -> np.ndarray:
    if draws.shape[0] != 2 or draws.shape[1] % 2:
        raise ValueError("Expected two equal, even-length chains")
    midpoint = draws.shape[1] // 2
    return np.concatenate((draws[:, :midpoint], draws[:, midpoint:]), axis=0)


def rank_normalize(draws: np.ndarray) -> np.ndarray:
    flat = draws.reshape(draws.shape[0] * draws.shape[1], -1)
    ranks = rankdata(flat, axis=0, method="average")
    z = ndtri((ranks - 0.375) / (len(flat) + 0.25))
    return z.reshape(draws.shape)


def diagnostic_arrays(draws: np.ndarray, batch_size: int = 16) -> dict:
    """Rank/folded split R-hat and bulk/tail ESS, one parameter batch at a time."""
    original_shape = draws.shape[2:]
    flat = draws.reshape(draws.shape[0], draws.shape[1], -1)
    results = {
        name: np.full(flat.shape[-1], np.nan)
        for name in (
            "rhat",
            "bulk_ess",
            "tail_ess",
        )
    }
    for start in range(0, flat.shape[-1], batch_size):
        stop = min(start + batch_size, flat.shape[-1])
        split = split_chains(flat[:, :, start:stop].astype(np.float64))
        spread = np.ptp(split, axis=(0, 1))
        active = spread > 1e-12
        if not np.any(active):
            continue
        split = split[:, :, active]
        z = rank_normalize(split)
        folded = rank_normalize(np.abs(split - np.median(split, axis=(0, 1))))
        with np.errstate(divide="ignore", invalid="ignore"):
            rhat = np.maximum(gelman_rubin(z), gelman_rubin(folded))
            bulk = effective_sample_size(z)
            q05, q95 = np.quantile(split, (0.05, 0.95), axis=(0, 1))
            lower = effective_sample_size((split <= q05).astype(np.float64))
            upper = effective_sample_size((split >= q95).astype(np.float64))
            tail = np.minimum(lower, upper)
        indices = np.arange(start, stop)[active]
        results["rhat"][indices] = rhat
        results["bulk_ess"][indices] = bulk
        results["tail_ess"][indices] = tail
    return {key: value.reshape(original_shape) for key, value in results.items()}


def classical_rhat_from_sums(
    sums: np.ndarray,
    squares: np.ndarray,
    half_length: int,
) -> np.ndarray:
    means = sums / half_length
    variances = (squares - sums * means) / (half_length - 1)
    within = variances.mean(axis=(0, 1))
    between = half_length * means.reshape(4, *means.shape[2:]).var(
        axis=0,
        ddof=1,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(((half_length - 1) * within + between) / (half_length * within))


def load_chain(
    root: Path,
    sampled_people: np.ndarray | None,
) -> tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray, dict]:
    config = json.loads((root / "config.json").read_text())
    manifest_path = root / "chain_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verified = json.loads((root / "MIXTURE_CHAIN_VERIFIED.json").read_text())
    completed = json.loads((root / "MIXTURE_CHAIN_COMPLETED.json").read_text())
    if (root / "MIXTURE_CHAIN_FAILED.json").exists():
        raise ValueError(f"Failure marker in {root}")
    if (
        verified["status"] != "verified"
        or verified["chain_manifest_sha256"] != sha256_file(manifest_path)
        or completed["chain_manifest_sha256"] != sha256_file(manifest_path)
        or config["samples"] != SAMPLES
        or config["warmup"] != WARMUP
        or config["chunk_size"] != CHUNK_SIZE
        or config["global_archive_stride"] != GLOBAL_STRIDE
        or config["archive_stride"] != PERSON_STRIDE
        or len(manifest["chunks"]) != CHUNKS
    ):
        raise ValueError(f"Unverified or incompatible chain {root}")
    people = config["source_people"]
    if sampled_people is None:
        seed = 20260922 + config["configuration"]["configuration_index"]
        sampled_people = np.sort(
            np.random.default_rng(seed).choice(people, min(512, people), replace=False)
        )
    globals_: dict[str, list[np.ndarray]] = {}
    halves = np.zeros((2, people, 4), dtype=np.float64)
    half_squares = np.zeros_like(halves)
    selected = np.empty(
        (PERSON_ARCHIVED, len(sampled_people), 4), dtype=np.float32
    )
    for number, entry in enumerate(manifest["chunks"]):
        path = root / f"chunk-{number:03d}.npz"
        if (
            entry["number"] != number
            or entry["draw_start"] != number * CHUNK_SIZE
            or entry["draw_stop"] != (number + 1) * CHUNK_SIZE
            or path.stat().st_size != entry["bytes"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError(f"Chunk binding failed: {path}")
        with np.load(path, allow_pickle=False) as archive:
            index = archive["person_draw_index"]
            expected = np.arange(
                number * CHUNK_SIZE + PERSON_STRIDE - 1,
                (number + 1) * CHUNK_SIZE,
                PERSON_STRIDE,
            )
            if not np.array_equal(index, expected):
                raise ValueError(f"Misaligned respondent archive: {path}")
            person = archive["person_archive"]
            if (
                person.shape != (PERSON_ARCHIVE_PER_CHUNK, people, 4)
                or person.dtype != np.float32
            ):
                raise ValueError(f"Malformed respondent archive: {path}")
            half = number // (CHUNKS // 2)
            halves[half] += person.sum(axis=0, dtype=np.float64)
            half_squares[half] += np.square(person.astype(np.float64)).sum(axis=0)
            selected[
                number * PERSON_ARCHIVE_PER_CHUNK:
                (number + 1) * PERSON_ARCHIVE_PER_CHUNK
            ] = person[:, sampled_people]

            weights = archive["weights"].astype(np.float64)
            component = archive["component_mean"].astype(np.float64)
            mixture_mean = np.einsum("dk,dkp->dp", weights, component)
            variance = np.einsum(
                "dk,dkp->dp",
                weights,
                (component - mixture_mean[:, None]) ** 2,
            ) + np.diagonal(archive["Sigma_B"], axis1=1, axis2=2)
            derived = {
                "mixture_mean": mixture_mean,
                "mixture_sd": np.sqrt(np.maximum(variance, 0)),
                "sorted_weights": np.sort(weights, axis=1)[:, ::-1],
                "largest_weight": weights.max(axis=1),
                "weight_entropy": -(weights * np.log(np.maximum(weights, 1e-300))).sum(
                    axis=1
                ),
            }
            for name in (
                "mu_a",
                "sigma_a",
                "mu_B",
                "Gamma",
                "delta",
                "sigma_campaign",
                "sigma_ad",
                "metric_occupied_components",
                "metric_log_likelihood_per_observation",
            ):
                globals_.setdefault(name, []).append(archive[name])
            for name, value in derived.items():
                globals_.setdefault(name, []).append(value)
        if number % 100 == 99:
            print(
                f"{root.name}: verified and loaded {number + 1}/{CHUNKS} chunks",
                flush=True,
            )
    return (
        {key: np.concatenate(value) for key, value in globals_.items()},
        config,
        halves,
        half_squares,
        selected,
        {
            "chain_root": str(root),
            "chain_verified_sha256": sha256_file(root / "MIXTURE_CHAIN_VERIFIED.json"),
            "chain_manifest_sha256": sha256_file(manifest_path),
        },
    )


def run(run_root: Path, index: int, output_root: Path) -> dict:
    configuration = CONFIGURATIONS[index]
    chain_roots = [run_root / configuration / f"chain-{chain}" for chain in (0, 1)]
    first = load_chain(chain_roots[0], None)
    sampled_people = np.sort(
        np.random.default_rng(20260922 + index).choice(
            first[1]["source_people"],
            min(512, first[1]["source_people"]),
            replace=False,
        )
    )
    second = load_chain(chain_roots[1], sampled_people)
    if first[1]["configuration"] != second[1]["configuration"]:
        raise ValueError("Chain configurations disagree")
    traces = {}
    for name in first[0]:
        if first[0][name].shape != second[0][name].shape:
            raise ValueError(f"Global trace shape mismatch: {name}")
        traces[name] = np.stack((first[0][name], second[0][name]))
    global_results = {}
    for name, draws in traces.items():
        diagnostics = diagnostic_arrays(draws)
        global_results[name] = {
            "shape": list(draws.shape[2:]),
            "rhat": summarize(diagnostics["rhat"]),
            "bulk_ess": summarize(diagnostics["bulk_ess"]),
            "tail_ess": summarize(diagnostics["tail_ess"]),
            "rhat_gt_1_01": int(np.count_nonzero(diagnostics["rhat"] > 1.01)),
            "rhat_gt_1_05": int(np.count_nonzero(diagnostics["rhat"] > 1.05)),
        }
        if draws.ndim == 2:
            global_results[name]["posterior_median"] = float(np.median(draws))
    print("global diagnostics complete", flush=True)

    sums = np.stack((first[2], second[2]))
    squares = np.stack((first[3], second[3]))
    all_person_rhat = classical_rhat_from_sums(
        sums, squares, PERSON_ARCHIVED // 2
    )
    sampled_draws = np.stack((first[4], second[4]))
    sampled_diagnostics = diagnostic_arrays(sampled_draws)
    person_results = {}
    for position, name in enumerate(PERSON_NAMES):
        rhat = sampled_diagnostics["rhat"][:, position]
        person_results[name] = {
            "all_people_classical_split_rhat": summarize(all_person_rhat[:, position]),
            "sampled_people_rank_folded_split_rhat": summarize(rhat),
            "sampled_people_bulk_ess": summarize(
                sampled_diagnostics["bulk_ess"][:, position]
            ),
            "sampled_people_tail_ess": summarize(
                sampled_diagnostics["tail_ess"][:, position]
            ),
            "sampled_rhat_gt_1_01": int(np.count_nonzero(rhat > 1.01)),
            "sampled_rhat_gt_1_05": int(np.count_nonzero(rhat > 1.05)),
        }
    print("person diagnostics complete", flush=True)

    output = output_root / configuration
    output.mkdir(parents=True, exist_ok=False)
    arrays_path = output / "diagnostic_arrays.npz"
    np.savez_compressed(
        arrays_path,
        sampled_person_index=sampled_people,
        all_person_classical_split_rhat=all_person_rhat,
        sampled_person_rank_folded_split_rhat=sampled_diagnostics["rhat"],
        sampled_person_bulk_ess=sampled_diagnostics["bulk_ess"],
        sampled_person_tail_ess=sampled_diagnostics["tail_ess"],
    )
    result = {
        "schema_version": "pg-mixture-long-convergence-diagnostics-v1",
        "configuration": configuration,
        "chain_count": 2,
        "post_warmup_draws_per_chain": SAMPLES,
        "warmup_per_chain": WARMUP,
        "global_archive_stride": GLOBAL_STRIDE,
        "person_archive_stride": PERSON_STRIDE,
        "person_archived_draws_per_chain": PERSON_ARCHIVED,
        "person_count": first[1]["source_people"],
        "sampled_person_count_for_ess": len(sampled_people),
        "sampled_person_seed": 20260922 + index,
        "methods": {
            "global": (
                "label-invariant rank-normalized folded split R-hat and Geyer "
                "bulk/tail ESS on every-twentieth draw"
            ),
            "person_rhat": (
                "classical split R-hat for all persons; rank-normalized "
                "folded split R-hat for fixed random sample"
            ),
            "person_ess": (
                "rank-normalized bulk and 5%/95% tail ESS for fixed random "
                "sample on every-200th archive only"
            ),
            "mixture_labels": (
                "component order is ignored except for sorted weight order statistics"
            ),
        },
        "source_chains": [first[5], second[5]],
        "global": global_results,
        "person": person_results,
        "diagnostic_arrays_sha256": sha256_file(arrays_path),
    }
    json_path = output / "diagnostics.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    marker = {
        "schema_version": "pg-mixture-convergence-diagnostics-completed-v1",
        "status": "completed",
        "configuration": configuration,
        "diagnostics_sha256": sha256_file(json_path),
        "diagnostic_arrays_sha256": sha256_file(arrays_path),
    }
    (output / "DIAGNOSTICS_COMPLETED.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--index", type=int, choices=range(4), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.run_root, args.index, args.output_root)
    print(
        json.dumps(
            {
                "configuration": result["configuration"],
                "person": result["person"],
                "global_mu_a": result["global"]["mu_a"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
