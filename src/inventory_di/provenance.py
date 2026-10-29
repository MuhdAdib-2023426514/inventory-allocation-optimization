"""Stage lineage and effective configuration acceptance contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from inventory_di.io import get_environment_snapshot, save_json, sha256_file, source_provenance


def config_digest(cfg):
    return hashlib.sha256(json.dumps(cfg.model_dump(), sort_keys=True).encode()).hexdigest()


def stage_path(cfg, stage):
    base = cfg.data.processed_dir if stage == "prepare" else cfg.data.artifacts_dir
    return Path(base) / f"{stage}_manifest.json"


def record_stage(cfg, stage, inputs, outputs):
    data = dict(stage=stage, run_id=cfg.run_id, effective_config=cfg.model_dump(),
                effective_config_sha256=config_digest(cfg), source=source_provenance(),
                environment=get_environment_snapshot(),
                inputs={str(p): sha256_file(p) for p in inputs},
                outputs={str(p): sha256_file(p) for p in outputs})
    save_json(data, stage_path(cfg, stage))
    return data


def validate_stage(cfg, stage, check_source=True):
    path = stage_path(cfg, stage)
    if not path.exists():
        raise ValueError(f"Missing {path}; run {stage} for this config/profile")
    data = json.loads(path.read_text())
    if data["effective_config_sha256"] != config_digest(cfg):
        raise ValueError(f"Effective config mismatch in {path}; regenerate {stage}")
    if check_source:
        # Full source snapshots remain recorded. Infrastructure/presentation-only
        # edits do not require retraining unchanged scientific computations.
        current = source_provenance()["source_files"]
        critical = ("/models/", "/optimization/", "/simulation/", "/data/", "/evaluation/backtest.py", "/evaluation/metrics.py", "/evaluation/sensitivity.py", "/config.py")
        before = {k:v for k,v in data["source"]["source_files"].items() if any(part in k for part in critical)}
        after = {k:v for k,v in current.items() if any(part in k for part in critical)}
        if before != after:
            raise ValueError(f"Scientific source changed since {stage}; regenerate {stage}")
    for p, digest in {**data["inputs"], **data["outputs"]}.items():
        if not Path(p).exists() or sha256_file(p) != digest:
            raise ValueError(f"Stale or modified artifact {p}; regenerate {stage}")
    return data
