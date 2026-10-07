"""Summarize two archived PG classical-HB global chains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpyro.diagnostics import summary
from run_sparse_mixture_teacher import atomic_json

SITES = (
    "mu_a", "sigma_a", "mu_B", "Gamma_global", "Gamma_local",
    "Gamma", "Sigma_B", "delta", "sigma_campaign", "sigma_ad",
    "campaign_effect", "ad_effect",
)
ACCEPTANCE_SITES = (
    "metric_sigma_b_accepted", "metric_sigma_a_accepted",
    "metric_sigma_campaign_accepted", "metric_sigma_item_accepted",
    "metric_hs_global_acceptance", "metric_hs_local_acceptance",
    "metric_off_centered_scale_acceptance",
)


def read_chain(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    if not (root / "CLASSIC_CHAIN_COMPLETED.json").is_file():
        raise ValueError(f"Incomplete chain: {root}")
    config = json.loads((root / "config.json").read_text())
    manifest = json.loads((root / "chain_manifest.json").read_text())
    series = {name: [] for name in (*SITES, *ACCEPTANCE_SITES)}
    for entry in manifest["chunks"]:
        path = root / f"chunk-{entry['number']:03d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            for name in series:
                series[name].append(np.asarray(archive[name]))
    return config, {name: np.concatenate(parts) for name, parts in series.items()}


def diagnose(roots: list[Path]) -> dict:
    if len(roots) != 2:
        raise ValueError("Exactly two independent chains are required")
    loaded = [read_chain(root) for root in roots]
    first, second = loaded[0][0], loaded[1][0]
    if (
        first["configuration"]["configuration_id"]
        != second["configuration"]["configuration_id"]
        or first["model"] != second["model"]
        or first["mode"] != second["mode"]
        or first["chain"] == second["chain"]
        or first["seed"] == second["seed"]
    ):
        raise ValueError("Incompatible or duplicated chain identities")
    sites = {}
    for name in SITES:
        values = np.stack([chain[name] for _, chain in loaded])
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite global trace: {name}")
        record = summary({name: values}, group_by_chain=True)[name]
        ess = np.asarray(record["n_eff"])
        rhat = np.asarray(record["r_hat"])
        sites[name] = {
            "shape": list(values.shape[2:]),
            "ess_min": float(np.nanmin(ess)),
            "ess_median": float(np.nanmedian(ess)),
            "split_rhat_max": float(np.nanmax(rhat)),
            "split_rhat_median": float(np.nanmedian(rhat)),
            "finite_diagnostics": bool(
                np.isfinite(ess).all() and np.isfinite(rhat).all()
            ),
        }
    acceptance = {
        name.removeprefix("metric_"): float(np.mean([
            chain[name].mean() for _, chain in loaded
        ])) for name in ACCEPTANCE_SITES
    }
    return {
        "schema_version": "pg-classic-hb-global-diagnostics-v1",
        "configuration_id": first["configuration"]["configuration_id"],
        "mode": first["mode"], "model": first["model"],
        "chains": [first["chain"], second["chain"]],
        "retained_draws_per_chain": first["samples"],
        "archived_global_draws_per_chain": len(loaded[0][1]["mu_a"]),
        "diagnostic_method": (
            "NumPyro split R-hat and autocorrelation ESS on archived globals"
        ),
        "sites": sites, "mh_acceptance": acceptance,
        "convergence_certified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = diagnose(args.chain_root)
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
