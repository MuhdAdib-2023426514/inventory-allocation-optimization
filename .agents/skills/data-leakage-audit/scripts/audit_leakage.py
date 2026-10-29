#!/usr/bin/env python3
"""Deterministic temporal and target leakage auditor for inventory allocation.

Audits pipeline lineage manifests and scans Python source directories for violations
of AGENTS.md Sections 1.2, 4, 6, 7, 8, 9, and 14.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any


def parse_iso_date(val: str) -> date:
    """Parse YYYY-MM-DD date string."""
    return date.fromisoformat(str(val))


def audit_lineage_manifest(manifest: dict[str, Any]) -> list[str]:
    """Audit a dataset & pipeline lineage manifest for temporal leakage."""
    errors: list[str] = []
    required_top = [
        "records",
        "cohort_selection_end_date",
        "H_estimation_end_date",
        "copula_fit_end_date",
        "calibration_fit_splits",
        "proportional_baseline_max_lag_date_offset",
        "realized_test_demand_clipped",
    ]
    for key in required_top:
        if key not in manifest:
            errors.append(f"Missing required manifest field '{key}'")
    if errors:
        return errors

    records = manifest["records"]
    if not records:
        errors.append("Manifest 'records' list is empty")
        return errors

    split_origins: dict[str, set[date]] = {"train": set(), "val": set(), "test": set()}
    origin_splits: dict[date, set[str]] = {}

    for idx, rec in enumerate(records):
        try:
            origin = parse_iso_date(rec["forecast_origin"])
            t_start = parse_iso_date(rec["target_start"])
            t_end = parse_iso_date(rec["target_end"])
            f_max = parse_iso_date(rec["feature_max_date"])
            p_max = parse_iso_date(rec["price_max_date"])
        except (KeyError, ValueError) as exc:
            errors.append(f"Record {idx} invalid or missing date fields: {exc}")
            continue

        split = str(rec.get("split", "")).lower()
        if split not in split_origins:
            errors.append(f"Record {idx} has invalid split {split!r}")
            continue

        split_origins[split].add(origin)
        origin_splits.setdefault(origin, set()).add(split)

        # Causal feature ordering: feature_max_date <= forecast_origin < target_start <= target_end
        if f_max > origin:
            errors.append(
                f"Record {idx} feature leakage: feature_max_date {f_max} > forecast_origin {origin}"
            )
        if p_max > origin and not rec.get("scheduled_price_known_at_origin", False):
            errors.append(
                f"Record {idx} price leakage: unscheduled price_max_date {p_max} > forecast_origin {origin}"
            )
        if not (origin < t_start <= t_end):
            errors.append(
                f"Record {idx} invalid target window: origin={origin}, target=[{t_start}, {t_end}]"
            )
        if (t_end - t_start).days != 6:
            errors.append(
                f"Record {idx} target window is not 7 calendar days: {t_start} to {t_end}"
            )

    # Check whole-time-block splitting (no origin belongs to multiple splits)
    for origin_dt, splits_at_origin in origin_splits.items():
        if len(splits_at_origin) > 1:
            errors.append(
                f"Origin {origin_dt} appears in multiple splits {sorted(splits_at_origin)} (must split by whole time blocks)"
            )

    # Check chronological ordering train < val < test
    if split_origins["train"] and split_origins["val"]:
        if max(split_origins["train"]) >= min(split_origins["val"]):
            errors.append(
                f"Chronological overlap: max(train)={max(split_origins['train'])} >= min(val)={min(split_origins['val'])}"
            )
    if split_origins["val"] and split_origins["test"]:
        if max(split_origins["val"]) >= min(split_origins["test"]):
            errors.append(
                f"Chronological overlap: max(val)={max(split_origins['val'])} >= min(test)={min(split_origins['test'])}"
            )
    if split_origins["train"] and split_origins["test"]:
        if max(split_origins["train"]) >= min(split_origins["test"]):
            errors.append(
                f"Chronological overlap: max(train)={max(split_origins['train'])} >= min(test)={min(split_origins['test'])}"
            )

    if not split_origins["train"]:
        errors.append("No 'train' split origins found in manifest")
        return errors

    train_max_origin = max(split_origins["train"])
    # Remember target_end of the last training origin is train_max_origin + 7 days;
    # parameters fit on training window must not use dates after the first validation/test origin.
    holdout_Boundary = (
        min(split_origins["val"])
        if split_origins["val"]
        else (min(split_origins["test"]) if split_origins["test"] else train_max_origin)
    )

    for field_name in ("cohort_selection_end_date", "H_estimation_end_date", "copula_fit_end_date"):
        cutoff = parse_iso_date(manifest[field_name])
        if cutoff > holdout_Boundary:
            errors.append(
                f"Holdout leakage in {field_name}: cutoff {cutoff} > holdout boundary {holdout_Boundary}"
            )

    cal_splits = {str(s).lower() for s in manifest["calibration_fit_splits"]}
    if "test" in cal_splits:
        errors.append("Probability calibration fit on 'test' split")

    if int(manifest["proportional_baseline_max_lag_date_offset"]) > 0:
        errors.append(
            f"Proportional baseline uses future demand (offset={manifest['proportional_baseline_max_lag_date_offset']} > 0)"
        )

    if bool(manifest["realized_test_demand_clipped"]):
        errors.append("Realized test demand was clipped at PMF cap H")

    return errors


def scan_python_source_for_leakage(scan_dir: Path) -> list[str]:
    """Scan Python files under scan_dir for common temporal leakage anti-patterns."""
    errors: list[str] = []
    if not scan_dir.exists():
        return errors

    for py_file in sorted(scan_dir.rglob("*.py")):
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError as exc:
            errors.append(f"{py_file}: SyntaxError {exc}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                # Detect train_test_split without shuffle=False
                if func_name == "train_test_split":
                    shuffle_kw = next((kw for kw in node.keywords if kw.arg == "shuffle"), None)
                    if shuffle_kw is None or (
                        isinstance(shuffle_kw.value, ast.Constant)
                        and shuffle_kw.value.value is not False
                    ):
                        errors.append(
                            f"{py_file}:{node.lineno}: train_test_split called without shuffle=False on time-series data"
                        )
    return errors


def run_self_test() -> dict[str, Any]:
    """Execute built-in positive and negative leakage audit checks."""
    errors: list[str] = []

    valid_manifest = {
        "records": [
            {
                "forecast_origin": "2016-01-03",
                "target_start": "2016-01-04",
                "target_end": "2016-01-10",
                "feature_max_date": "2016-01-03",
                "price_max_date": "2016-01-03",
                "split": "train",
                "store_id": "CA_1",
                "sku_id": "HOUSEHOLD_1_001",
            },
            {
                "forecast_origin": "2016-01-17",
                "target_start": "2016-01-18",
                "target_end": "2016-01-24",
                "feature_max_date": "2016-01-17",
                "price_max_date": "2016-01-17",
                "split": "val",
                "store_id": "CA_1",
                "sku_id": "HOUSEHOLD_1_001",
            },
            {
                "forecast_origin": "2016-01-31",
                "target_start": "2016-02-01",
                "target_end": "2016-02-07",
                "feature_max_date": "2016-01-31",
                "price_max_date": "2016-01-31",
                "split": "test",
                "store_id": "CA_1",
                "sku_id": "HOUSEHOLD_1_001",
            },
        ],
        "cohort_selection_end_date": "2016-01-10",
        "H_estimation_end_date": "2016-01-10",
        "copula_fit_end_date": "2016-01-10",
        "calibration_fit_splits": ["val"],
        "proportional_baseline_max_lag_date_offset": 0,
        "realized_test_demand_clipped": False,
    }
    errors.extend(audit_lineage_manifest(valid_manifest))

    # Negative check 1: feature date after forecast_origin and copula fit on test
    leaked_manifest = json.loads(json.dumps(valid_manifest))
    leaked_manifest["records"][0]["feature_max_date"] = "2016-01-05"
    leaked_manifest["copula_fit_end_date"] = "2016-02-07"
    leaked_manifest["calibration_fit_splits"] = ["val", "test"]
    leaked_manifest["realized_test_demand_clipped"] = True
    leaked_errs = audit_lineage_manifest(leaked_manifest)

    for expected_substr in (
        "feature leakage",
        "copula_fit_end_date",
        "Probability calibration fit on 'test'",
        "clipped at PMF cap H",
    ):
        if not any(expected_substr in e for e in leaked_errs):
            errors.append(f"Self-test failed to detect leakage condition: {expected_substr}")

    # Negative check 2: mixed splits on same forecast_origin
    mixed_split_manifest = json.loads(json.dumps(valid_manifest))
    mixed_split_manifest["records"][1]["forecast_origin"] = "2016-01-03"
    mixed_split_manifest["records"][1]["target_start"] = "2016-01-04"
    mixed_split_manifest["records"][1]["target_end"] = "2016-01-10"
    mixed_split_manifest["records"][1]["feature_max_date"] = "2016-01-03"
    mixed_split_manifest["records"][1]["price_max_date"] = "2016-01-03"
    mixed_errs = audit_lineage_manifest(mixed_split_manifest)
    if not any("multiple splits" in e for e in mixed_errs):
        errors.append("Self-test failed to detect non-block random split at same origin")

    return {
        "skill": "data-leakage-audit",
        "status": "PASS" if not errors else "FAIL",
        "checks_run": [
            "valid_chronological_lineage_manifest",
            "negative_feature_and_copula_leakage_detection",
            "negative_test_calibration_and_demand_clipping_detection",
            "negative_same_origin_mixed_split_detection",
        ],
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Temporal and target leakage auditor.")
    parser.add_argument("--self-test", action="store_true", help="Run built-in leakage self-test.")
    parser.add_argument("--manifest", type=Path, help="Path to lineage manifest JSON.")
    parser.add_argument("--scan-dir", type=Path, help="Directory of Python files to AST-scan.")
    args = parser.parse_args(argv)

    if not args.self_test and not args.manifest and not args.scan_dir:
        parser.error("Provide --self-test, --manifest <path>, and/or --scan-dir <dir>")

    if args.self_test:
        report = run_self_test()
    else:
        errs: list[str] = []
        if args.manifest:
            data = json.loads(args.manifest.read_text(encoding="utf-8"))
            errs.extend(audit_lineage_manifest(data))
        if args.scan_dir:
            errs.extend(scan_python_source_for_leakage(args.scan_dir))
        report = {
            "skill": "data-leakage-audit",
            "status": "PASS" if not errs else "FAIL",
            "errors": errs,
        }

    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
