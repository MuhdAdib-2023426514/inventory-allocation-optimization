"""Command-line interface (`python -m inventory_di.cli ...`) (`AGENTS.md` §12)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from inventory_di.config import load_config
from inventory_di.data.m5 import check_m5_files_exist, ingest_m5_dataset
from inventory_di.data.synthetic import generate_synthetic_demo_dataset
from inventory_di.evaluation.backtest import (
    load_sku_forecast_artifacts,
    run_rolling_backtest,
    train_all_models,
)
from inventory_di.evaluation.validation import (
    build_dashboard_qa_bundle,
    generate_markdown_report,
)
from inventory_di.io import save_json, save_parquet
from inventory_di.provenance import record_stage, validate_stage


def cmd_doctor() -> int:
    from inventory_di.doctor import run_environment_doctor

    report = run_environment_doctor()
    print(json.dumps(report, indent=2))
    return 0


def cmd_prepare(config_path: str, quick: bool = False) -> int:
    cfg = load_config(config_path, quick=quick)
    proc_dir = Path(cfg.data.processed_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)

    if cfg.data_mode == "synthetic_demo":
        weekly_df, inv_df, lineage = generate_synthetic_demo_dataset(cfg)
    else:
        weekly_df, inv_df, lineage = ingest_m5_dataset(cfg)

    w_sha = save_parquet(weekly_df, proc_dir / "weekly_demand.parquet")
    i_sha = save_parquet(inv_df, proc_dir / "inventory_config.parquet")
    lineage["weekly_demand_sha256"] = w_sha
    lineage["inventory_config_sha256"] = i_sha
    save_json(lineage, proc_dir / "lineage_manifest.json")
    record_stage(cfg, "prepare", list(check_m5_files_exist(cfg.data.raw_m5_dir).values()) if cfg.mode == "m5" else [], [proc_dir / name for name in
                 ("weekly_demand.parquet", "inventory_config.parquet", "lineage_manifest.json")])

    print(
        json.dumps(
            {
                "command": "prepare",
                "mode": cfg.mode,
                "data_mode": cfg.data_mode,
                "processed_dir": str(proc_dir),
                "weekly_demand_rows": len(weekly_df),
                "inventory_config_rows": len(inv_df),
                "weekly_demand_sha256": w_sha,
            },
            indent=2,
        )
    )
    return 0


def cmd_train(config_path: str, model_choice: str = "all", quick: bool = False) -> int:
    cfg = load_config(config_path, quick=quick)
    if model_choice == "all" and (Path(cfg.data.artifacts_dir) / "train_manifest.json").exists():
        try:
            validate_stage(cfg, "train")
            print(json.dumps({"command": "train", "run_id": cfg.run_id, "status": "validated_cached_stage"}))
            return 0
        except ValueError:
            pass
    proc_dir = Path(cfg.data.processed_dir)
    w_path = proc_dir / "weekly_demand.parquet"
    i_path = proc_dir / "inventory_config.parquet"
    l_path = proc_dir / "lineage_manifest.json"
    if not w_path.exists() or not i_path.exists():
        cmd_prepare(config_path, quick=quick)

    weekly_df = pd.read_parquet(w_path)
    inv_df = pd.read_parquet(i_path)
    lineage = json.loads(l_path.read_text(encoding="utf-8")) if l_path.exists() else {}

    validate_stage(cfg, "prepare")
    if model_choice != "all" and (Path(cfg.data.artifacts_dir) / "train_manifest.json").exists():
        validate_stage(cfg, "train")
    diag = train_all_models(cfg, weekly_df, inv_df, lineage, model_choice=model_choice)
    art_dir = Path(cfg.data.artifacts_dir)
    checkpoints = [p for p in [art_dir / "supervised_models.pt", art_dir / "dfl_models.pt"] if p.exists()]
    record_stage(cfg, "train", [w_path, i_path, l_path], checkpoints + [art_dir / "training_diagnostics.json"])
    print(
        json.dumps(
            {
                "command": "train",
                "mode": cfg.mode,
                "data_mode": cfg.data_mode,
                "model_choice": model_choice,
                "checkpoints": diag["checkpoints"],
            },
            indent=2,
        )
    )
    return 0


def cmd_backtest(config_path: str, quick: bool = False) -> int:
    cfg = load_config(config_path, quick=quick)
    if (Path(cfg.data.artifacts_dir) / "backtest_manifest.json").exists():
        try:
            validate_stage(cfg, "backtest")
            print(json.dumps({"command": "backtest", "run_id": cfg.run_id, "status": "validated_cached_stage"}))
            return 0
        except ValueError:
            pass
    proc_dir = Path(cfg.data.processed_dir)
    art_dir = Path(cfg.data.artifacts_dir)

    w_path = proc_dir / "weekly_demand.parquet"
    i_path = proc_dir / "inventory_config.parquet"
    l_path = proc_dir / "lineage_manifest.json"
    if not w_path.exists() or not i_path.exists():
        cmd_prepare(config_path, quick=quick)

    sup_ckpt = art_dir / "supervised_models.pt"
    dfl_ckpt = art_dir / "dfl_models.pt"
    if not sup_ckpt.exists() or not dfl_ckpt.exists():
        cmd_train(config_path, model_choice="all", quick=quick)

    weekly_df = pd.read_parquet(w_path)
    inv_df = pd.read_parquet(i_path)
    lineage = json.loads(l_path.read_text(encoding="utf-8")) if l_path.exists() else {}

    validate_stage(cfg, "prepare")
    validate_stage(cfg, "train")
    sup_artifacts = load_sku_forecast_artifacts(sup_ckpt, hidden_dim=cfg.model.hidden_dim)
    dfl_artifacts = load_sku_forecast_artifacts(dfl_ckpt, hidden_dim=cfg.model.hidden_dim)

    res = run_rolling_backtest(cfg, weekly_df, inv_df, sup_artifacts, dfl_artifacts, lineage)
    record_stage(cfg, "backtest", [w_path, i_path, l_path, sup_ckpt, dfl_ckpt],
                 list(art_dir.glob("*.parquet")) + [art_dir / "run_manifest.json", art_dir / "cached_origin_bundles.json"])
    summary_df = res["summary_metrics"]
    print(
        json.dumps(
            {
                "command": "backtest",
                "mode": cfg.mode,
                "data_mode": cfg.data_mode,
                "num_test_origins": res["run_manifest"]["num_test_origins"],
                "policies": summary_df[["policy", "mean_cost_mu", "mean_regret_mu", "mean_fill_rate"]].to_dict(
                    orient="records"
                ),
            },
            indent=2,
        )
    )
    return 0


def cmd_report(config_path: str, quick: bool = False) -> int:
    cfg = load_config(config_path, quick=quick)
    art_dir = Path(cfg.data.artifacts_dir)
    sum_path = art_dir / "summary_metrics.parquet"
    sens_path = art_dir / "risk_sensitivity.parquet"
    diag_path = art_dir / "training_diagnostics.json"
    man_path = art_dir / "run_manifest.json"

    if not sum_path.exists() or not man_path.exists():
        cmd_backtest(config_path, quick=quick)

    validate_stage(cfg, "backtest")
    summary_df = pd.read_parquet(sum_path)
    sens_df = pd.read_parquet(sens_path) if sens_path.exists() else pd.DataFrame()
    training_diag = json.loads(diag_path.read_text(encoding="utf-8")) if diag_path.exists() else {}
    run_manifest = json.loads(man_path.read_text(encoding="utf-8"))

    generate_markdown_report(cfg, summary_df, training_diag, run_manifest)
    qa_bundle = build_dashboard_qa_bundle(cfg, summary_df, sens_df, run_manifest)
    save_json(qa_bundle, art_dir / "dashboard_qa_bundle.json")

    print(
        json.dumps(
            {
                "command": "report",
                "mode": cfg.mode,
                "data_mode": cfg.data_mode,
                "report_path": str(Path(cfg.data.reports_dir) / "results_summary.md"),
                "run_id": cfg.run_id,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inventory_di",
        description="Decision-Focused and Distributionally Robust Inventory Allocation CLI",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("doctor", help="Verify environment, solver, and PyEPO SPO+ gradient smoke test")

    for name, help_text in [
        ("make-demo", "Generate synthetic demo dataset and inventory config"),
        ("prepare", "Prepare weekly demand and inventory config contracts"),
        ("backtest", "Run 7-policy rolling-origin backtest and sensitivity sweeps"),
        ("report", "Export results_summary.md and dashboard QA bundle"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", required=True, help="Path to YAML config file")
        p.add_argument("--quick", action="store_true", help="Use quick smoke budget")

    p_train = sub.add_parser("train", help="Train supervised NLL and DFL (NLL+SPO+) models")
    p_train.add_argument("--config", required=True, help="Path to YAML config file")
    p_train.add_argument(
        "--model",
        default="all",
        choices=["all", "supervised", "dfl"],
        help="Which model family to train",
    )
    p_train.add_argument("--quick", action="store_true", help="Use quick smoke budget")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.subcommand == "doctor":
            return cmd_doctor()
        if args.subcommand in {"make-demo", "prepare"}:
            return cmd_prepare(args.config, quick=args.quick)
        if args.subcommand == "train":
            return cmd_train(args.config, model_choice=args.model, quick=args.quick)
        if args.subcommand == "backtest":
            return cmd_backtest(args.config, quick=args.quick)
        if args.subcommand == "report":
            return cmd_report(args.config, quick=args.quick)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
