"""Two-chain global and sampled-person diagnostics for PG classical HB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_sparse_mixture_long import (
    classical_rhat_from_sums,
    diagnostic_arrays,
    sha256_file,
    summarize,
)


CONFIGURATIONS = (
    "global-exposure80-seed2026090902",
    "within-campaign-last20pct-weeks-v1",
    "within-campaign-one-ad-seed2026091103",
    "hybrid-one-ad-plus-known-exposure20-seed2026091504",
)
SITES = (
    "mu_a", "sigma_a", "mu_B", "Gamma_global", "Gamma_local",
    "Gamma", "Sigma_B", "delta", "sigma_campaign", "sigma_ad",
    "campaign_effect", "ad_effect",
)
ACCEPTANCE = (
    "metric_sigma_b_accepted", "metric_sigma_a_accepted",
    "metric_sigma_campaign_accepted", "metric_sigma_item_accepted",
    "metric_hs_global_acceptance", "metric_hs_local_acceptance",
    "metric_off_centered_scale_acceptance",
)
PERSON_NAMES = ("a_i", "beta_brand", "beta_performance", "beta_duration30")
SAMPLES = 1_000_000
CHUNK_SIZE = 1_000
GLOBAL_STRIDE = 20
PERSON_STRIDE = 1_000
PERSON_ARCHIVED = SAMPLES // PERSON_STRIDE


def load_chain(
    root: Path, sampled_people: np.ndarray | None, seed: int
) -> tuple[dict[str, np.ndarray], dict, np.ndarray, np.ndarray, np.ndarray, dict]:
    config = json.loads((root / "config.json").read_text())
    manifest_path = root / "chain_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    marker = json.loads((root / "CLASSIC_CHAIN_VERIFIED.json").read_text())
    if (
        marker["status"] != "verified"
        or marker["chain_manifest_sha256"] != sha256_file(manifest_path)
        or config["mode"] != "production"
        or config["samples"] != SAMPLES
        or config["warmup"] != 20_000
        or config["chunk_size"] != CHUNK_SIZE
        or config["global_archive_stride"] != GLOBAL_STRIDE
        or config["person_archive_stride"] != PERSON_STRIDE
        or len(manifest["chunks"]) != SAMPLES // CHUNK_SIZE
    ):
        raise ValueError(f"Classical-HB chain provenance failed: {root}")
    people = config["source_people"]
    if sampled_people is None:
        sampled_people = np.sort(
            np.random.default_rng(seed).choice(
                people, min(512, people), replace=False
            )
        )
    traces = {name: [] for name in SITES}
    halves = np.zeros((2, people, 4), dtype=np.float64)
    half_squares = np.zeros_like(halves)
    selected = np.empty(
        (PERSON_ARCHIVED, len(sampled_people), 4), dtype=np.float32
    )
    accept_sums = {
        name: None for name in ACCEPTANCE
    }
    for number, entry in enumerate(manifest["chunks"]):
        path = root / f"chunk-{number:03d}.npz"
        if (
            entry["number"] != number
            or entry["draw_start"] != number * CHUNK_SIZE
            or entry["draw_stop"] != (number + 1) * CHUNK_SIZE
            or path.stat().st_size != entry["bytes"]
        ):
            raise ValueError(f"Chunk accounting failed: {path}")
        with np.load(path, allow_pickle=False) as archive:
            person_index = np.asarray(archive["person_draw_index"])
            global_index = np.asarray(archive["global_draw_index"])
            if (
                not np.array_equal(
                    person_index,
                    np.array([(number + 1) * CHUNK_SIZE - 1]),
                )
                or not np.array_equal(
                    global_index,
                    np.arange(
                        number * CHUNK_SIZE + GLOBAL_STRIDE - 1,
                        (number + 1) * CHUNK_SIZE,
                        GLOBAL_STRIDE,
                    ),
                )
            ):
                raise ValueError(f"Archived draw indices are misaligned: {path}")
            person = np.asarray(archive["person_archive"], dtype=np.float32)
            if person.shape != (1, people, 4):
                raise ValueError(f"Person archive shape failed: {path}")
            flat_person = person[0].astype(np.float64)
            half = number // (PERSON_ARCHIVED // 2)
            halves[half] += flat_person
            half_squares[half] += np.square(flat_person)
            selected[number] = person[0, sampled_people]
            for name in SITES:
                traces[name].append(np.asarray(archive[name]))
            for name in ACCEPTANCE:
                value = np.asarray(archive[name], dtype=np.float64).mean(axis=0)
                accept_sums[name] = (
                    value if accept_sums[name] is None
                    else accept_sums[name] + value
                )
        if number % 100 == 99:
            print(f"{root.name}: loaded {number + 1}/1000 chunks", flush=True)
    traces = {name: np.concatenate(parts) for name, parts in traces.items()}
    acceptance = {
        name: np.asarray(total / len(manifest["chunks"])).tolist()
        for name, total in accept_sums.items()
    }
    source = {
        "chain_root": str(root),
        "chain_manifest_sha256": sha256_file(manifest_path),
        "verification_sha256": sha256_file(root / "CLASSIC_CHAIN_VERIFIED.json"),
    }
    return traces, config, halves, half_squares, selected, {
        "source": source, "acceptance": acceptance,
    }


def run(run_root: Path, index: int, output_root: Path) -> dict:
    config_id = CONFIGURATIONS[index]
    seed = 20260923 + index
    roots = [run_root / config_id / f"chain-{chain}" for chain in (0, 1)]
    first = load_chain(roots[0], None, seed)
    selected_people = np.sort(
        np.random.default_rng(seed).choice(
            first[1]["source_people"],
            min(512, first[1]["source_people"]),
            replace=False,
        )
    )
    second = load_chain(roots[1], selected_people, seed)
    if (
        first[1]["configuration"] != second[1]["configuration"]
        or first[1]["chain"] == second[1]["chain"]
        or first[1]["seed"] == second[1]["seed"]
    ):
        raise ValueError("Independent classical-HB chains do not match")
    globals_ = {}
    for name in SITES:
        values = np.stack((first[0][name], second[0][name]))
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite classical-HB trace: {name}")
        diagnostics = diagnostic_arrays(values)
        globals_[name] = {
            "shape": list(values.shape[2:]),
            "rhat": summarize(diagnostics["rhat"]),
            "bulk_ess": summarize(diagnostics["bulk_ess"]),
            "tail_ess": summarize(diagnostics["tail_ess"]),
            "rhat_gt_1_01": int(np.count_nonzero(
                diagnostics["rhat"] > 1.01
            )),
            "rhat_gt_1_05": int(np.count_nonzero(
                diagnostics["rhat"] > 1.05
            )),
        }
    person_sums = np.stack((first[2], second[2]))
    person_squares = np.stack((first[3], second[3]))
    all_rhat = classical_rhat_from_sums(
        person_sums, person_squares, PERSON_ARCHIVED // 2
    )
    selected_draws = np.stack((first[4], second[4]))
    selected_diagnostics = diagnostic_arrays(selected_draws)
    person = {}
    for column, name in enumerate(PERSON_NAMES):
        person[name] = {
            "all_people_classical_split_rhat": summarize(all_rhat[:, column]),
            "sampled_rank_folded_split_rhat": summarize(
                selected_diagnostics["rhat"][:, column]
            ),
            "sampled_bulk_ess": summarize(
                selected_diagnostics["bulk_ess"][:, column]
            ),
            "sampled_tail_ess": summarize(
                selected_diagnostics["tail_ess"][:, column]
            ),
        }
    output = output_root / config_id
    output.mkdir(parents=True, exist_ok=False)
    arrays_path = output / "person_diagnostic_arrays.npz"
    np.savez_compressed(
        arrays_path,
        sampled_person_index=selected_people,
        all_person_classical_split_rhat=all_rhat,
        sampled_person_rank_folded_split_rhat=selected_diagnostics["rhat"],
        sampled_person_bulk_ess=selected_diagnostics["bulk_ess"],
        sampled_person_tail_ess=selected_diagnostics["tail_ess"],
    )
    result = {
        "schema_version": "pg-classic-hb-long-diagnostics-v1",
        "configuration_id": config_id,
        "retained_draws_per_chain": SAMPLES,
        "global_archive_stride": GLOBAL_STRIDE,
        "person_archive_stride": PERSON_STRIDE,
        "archived_global_draws_per_chain": SAMPLES // GLOBAL_STRIDE,
        "archived_person_draws_per_chain": PERSON_ARCHIVED,
        "sampled_people": len(selected_people),
        "global": globals_,
        "person": person,
        "mh_acceptance": {
            name.removeprefix("metric_"): (
                0.5 * (
                    np.asarray(first[5]["acceptance"][name])
                    + np.asarray(second[5]["acceptance"][name])
                )
            ).tolist()
            for name in ACCEPTANCE
        },
        "source_chains": [first[5]["source"], second[5]["source"]],
        "person_arrays_sha256": sha256_file(arrays_path),
        "convergence_certified": False,
    }
    result_path = output / "diagnostics.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "DIAGNOSTICS_COMPLETED.json").write_text(
        json.dumps({
            "status": "complete",
            "diagnostics_sha256": sha256_file(result_path),
            "person_arrays_sha256": sha256_file(arrays_path),
        }, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "configuration_id": config_id,
        "mu_a": globals_["mu_a"],
        "sigma_a": globals_["sigma_a"],
        "beta_brand": person["beta_brand"],
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
