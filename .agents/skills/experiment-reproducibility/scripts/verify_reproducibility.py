#!/usr/bin/env python3
"""Deterministic experiment reproducibility and provenance verifier.

Validates experiment run manifests, environment snapshots, data_mode isolation,
7-policy apples-to-apples parity, solver status logging, and bootstrap unit rules
in accordance with AGENTS.md Sections 1, 4, 8, 9, 12, 14, and 15.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

REQUIRED_POLICIES = {
    "equal_allocation",
    "proportional_allocation",
    "supervised_pto",
    "supervised_dro",
    "dfl_pto",
    "dfl_dro",
    "clairvoyant_oracle",
}

REQUIRED_METRICS = {
    "total_cost_mu",
    "shortage_cost_mu",
    "holding_cost_mu",
    "shipping_cost_mu",
    "filled_units",
    "unmet_units",
    "fill_rate",
    "oracle_regret_mu",
    "solver_status",
    "runtime_seconds",
}

VALID_DATA_MODES = {"synthetic_demo", "m5_sales_proxy"}


def collect_environment_snapshot() -> dict[str, Any]:
    """Inspect current local Python runtime and key research packages."""
    packages = [
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "torch",
        "pyepo",
        "pyomo",
        "highspy",
        "scipy",
        "plotly",
        "streamlit",
        "pydantic",
        "pytest",
        "yaml",
    ]
    installed: dict[str, bool] = {}
    for pkg in packages:
        mod_name = "sklearn" if pkg == "scikit-learn" else pkg
        installed[pkg] = importlib.util.find_spec(mod_name) is not None

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages_installed": installed,
    }


def verify_run_manifest(manifest: dict[str, Any]) -> list[str]:
    """Validate an experiment run manifest against AGENTS.md reproducibility gates."""
    errors: list[str] = []
    required_fields = [
        "run_id",
        "git_commit",
        "python_version",
        "platform",
        "global_seed",
        "config_path",
        "config_sha256",
        "data_mode",
        "economic_currency",
        "inventory_data_provenance",
        "dataset_hashes",
        "model_checkpoints",
        "test_origins",
        "common_scenario_seed",
        "num_scenarios",
        "policies_evaluated",
        "policy_origin_Fingerprints",
        "solver_statuses",
        "metrics_recorded",
        "bootstrap_unit",
        "small_origin_caveat_noted",
    ]
    for field in required_fields:
        if field not in manifest:
            errors.append(f"Missing required manifest field '{field}'")
    if errors:
        return errors

    # 1. Seed & Determinism
    if not isinstance(manifest["global_seed"], int) or isinstance(manifest["global_seed"], bool):
        errors.append(f"global_seed must be an integer, got {manifest['global_seed']!r}")
    if not isinstance(manifest["common_scenario_seed"], int):
        errors.append("common_scenario_seed must be an integer")

    # 2. Data Mode & Provenance Isolation
    data_mode = str(manifest["data_mode"])
    if data_mode not in VALID_DATA_MODES:
        errors.append(f"Invalid data_mode {data_mode!r}; must be one of {sorted(VALID_DATA_MODES)}")

    if str(manifest["economic_currency"]) != "MU":
        errors.append(
            f"economic_currency must be 'MU' (generic monetary units), got {manifest['economic_currency']!r}"
        )
    if str(manifest["inventory_data_provenance"]) != "synthetic":
        errors.append(
            f"inventory_data_provenance must be 'synthetic', got {manifest['inventory_data_provenance']!r}"
        )

    dataset_hashes = manifest.get("dataset_hashes", {})
    if not isinstance(dataset_hashes, dict) or not dataset_hashes:
        errors.append("dataset_hashes must be a non-empty dictionary of file->sha256")
    elif data_mode == "m5_sales_proxy":
        has_sales = (
            "sales_train_evaluation.csv" in dataset_hashes
            or "sales_train_validation.csv" in dataset_hashes
        )
        has_cal = "calendar.csv" in dataset_hashes
        has_prices = "sell_prices.csv" in dataset_hashes
        if not (has_sales and has_cal and has_prices):
            errors.append(
                "data_mode='m5_sales_proxy' requires raw M5 file hashes (sales_train_*, calendar.csv, sell_prices.csv)"
            )

    # 3. Model Checkpoints
    checkpoints = manifest.get("model_checkpoints", {})
    for ckpt_key in ("supervised_pmf", "dfl_pmf"):
        if ckpt_key not in checkpoints:
            errors.append(f"Missing required model checkpoint entry '{ckpt_key}'")
        else:
            ckpt = checkpoints[ckpt_key]
            if not ckpt.get("path") or not ckpt.get("sha256"):
                errors.append(f"Checkpoint '{ckpt_key}' must record non-empty 'path' and 'sha256'")

    # 4. Apples-to-apples 7-policy completeness & environment fingerprint parity
    evaluated = set(manifest["policies_evaluated"])
    missing_policies = REQUIRED_POLICIES - evaluated
    if missing_policies:
        errors.append(f"Missing required policies: {sorted(missing_policies)}")

    fingerprints = manifest.get("policy_origin_Fingerprints", {})
    fp_values = {fingerprints.get(pol) for pol in REQUIRED_POLICIES if pol in fingerprints}
    if len(fp_values) > 1 or None in fp_values:
        errors.append(
            f"Apples-to-apples violation: policies evaluated on mismatched environment fingerprints {fingerprints}"
        )

    # 5. Solver statuses
    statuses = manifest.get("solver_statuses", {})
    for solver_pol in (
        "supervised_pto",
        "supervised_dro",
        "dfl_pto",
        "dfl_dro",
        "clairvoyant_oracle",
    ):
        pol_statuses = statuses.get(solver_pol, [])
        if not pol_statuses:
            errors.append(f"No solver_statuses recorded for optimization policy '{solver_pol}'")
        elif any(str(s).lower() not in {"optimal", "ok", "optimal_solution"} for s in pol_statuses):
            errors.append(
                f"Non-optimal solver status recorded for '{solver_pol}': {pol_statuses}"
            )

    # 6. Metrics & Origin-wise Block Bootstrap
    recorded_metrics = set(manifest.get("metrics_recorded", []))
    missing_metrics = REQUIRED_METRICS - recorded_metrics
    if missing_metrics:
        errors.append(f"Missing required evaluation metrics: {sorted(missing_metrics)}")

    if str(manifest.get("bootstrap_unit")) != "forecast_origin":
        errors.append(
            f"bootstrap_unit must be 'forecast_origin' (not correlated item rows), got {manifest.get('bootstrap_unit')!r}"
        )

    test_origins = manifest.get("test_origins", [])
    if len(test_origins) < 20 and not bool(manifest.get("small_origin_caveat_noted")):
        errors.append(
            f"Run has only {len(test_origins)} test origins (<20) but small_origin_caveat_noted is False"
        )

    return errors


def run_self_test() -> dict[str, Any]:
    """Execute built-in positive and negative reproducibility manifest tests."""
    errors: list[str] = []

    valid_manifest = {
        "run_id": "demo_20261029_seed42",
        "git_commit": "uncommitted",
        "git_dirty": True,
        "python_version": "3.14.0",
        "platform": "macOS",
        "global_seed": 42,
        "config_path": "configs/demo.yaml",
        "config_sha256": "e3b0c44298fc1c149afbf4c8996fb924",
        "data_mode": "synthetic_demo",
        "economic_currency": "MU",
        "inventory_data_provenance": "synthetic",
        "dataset_hashes": {"weekly_demand.parquet": "a1b2c3d4"},
        "model_checkpoints": {
            "supervised_pmf": {"path": "artifacts/demo/sup.pt", "sha256": "1111aaaa"},
            "dfl_pmf": {"path": "artifacts/demo/dfl.pt", "sha256": "2222bbbb"},
        },
        "test_origins": ["2016-01-31", "2016-02-07", "2016-02-14", "2016-02-21"],
        "common_scenario_seed": 1001,
        "num_scenarios": 50,
        "policies_evaluated": sorted(REQUIRED_POLICIES),
        "policy_origin_Fingerprints": {pol: "fp_env_01" for pol in REQUIRED_POLICIES},
        "solver_statuses": {
            pol: ["optimal", "optimal", "optimal", "optimal"]
            for pol in (
                "supervised_pto",
                "supervised_dro",
                "dfl_pto",
                "dfl_dro",
                "clairvoyant_oracle",
            )
        },
        "metrics_recorded": sorted(REQUIRED_METRICS | {"cvar_90_cost_mu"}),
        "bootstrap_unit": "forecast_origin",
        "small_origin_caveat_noted": True,
    }
    errors.extend(verify_run_manifest(valid_manifest))

    # Negative check 1: claiming m5_sales_proxy without M5 raw file hashes + using RM currency
    fake_m5 = json.loads(json.dumps(valid_manifest))
    fake_m5["data_mode"] = "m5_sales_proxy"
    fake_m5["economic_currency"] = "RM"
    fake_m5["bootstrap_unit"] = "item_store_row"
    fake_m5_errs = verify_run_manifest(fake_m5)
    for expected in ("m5_sales_proxy", "economic_currency", "bootstrap_unit"):
        if not any(expected in e for e in fake_m5_errs):
            errors.append(f"Self-test failed to catch reproducibility violation: {expected}")

    # Negative check 2: mismatched environment fingerprints across policies
    mismatch_env = json.loads(json.dumps(valid_manifest))
    mismatch_env["policy_origin_Fingerprints"]["dfl_dro"] = "fp_env_DIFFERENT"
    if not any("Apples-to-apples violation" in e for e in verify_run_manifest(mismatch_env)):
        errors.append("Self-test failed to detect mismatched policy environment fingerprints")

    return {
        "skill": "experiment-reproducibility",
        "status": "PASS" if not errors else "FAIL",
        "environment": collect_environment_snapshot(),
        "checks_run": [
            "valid_demo_run_manifest",
            "negative_fake_m5_and_currency_detection",
            "negative_bootstrap_unit_detection",
            "negative_policy_environment_mismatch_detection",
        ],
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experiment reproducibility verifier.")
    parser.add_argument("--self-test", action="store_true", help="Run built-in self-test.")
    parser.add_argument("--check-env", action="store_true", help="Print local environment status.")
    parser.add_argument("--manifest", type=Path, help="Path to run manifest JSON.")
    args = parser.parse_args(argv)

    if not (args.self_test or args.check_env or args.manifest):
        parser.error("Provide --self-test, --check-env, or --manifest <path>")

    if args.self_test:
        report = run_self_test()
    elif args.check_env and not args.manifest:
        report = {
            "skill": "experiment-reproducibility",
            "status": "PASS",
            "environment": collect_environment_snapshot(),
        }
    else:
        data = json.loads(args.manifest.read_text(encoding="utf-8"))
        errs = verify_run_manifest(data)
        report = {
            "skill": "experiment-reproducibility",
            "manifest": str(args.manifest),
            "status": "PASS" if not errs else "FAIL",
            "errors": errs,
        }

    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
