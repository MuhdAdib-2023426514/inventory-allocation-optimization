"""Run actual project skill verifiers and retain complete outputs as release evidence."""

import json
import subprocess
import sys
from pathlib import Path

from inventory_di.config import load_config


def main():
    commands = [[sys.executable, "scripts/validate_skills.py"]]
    scripts = {"optimization-verification":"verify_optimization.py",
               "decision-focused-learning":"verify_dfl_contract.py",
               "data-leakage-audit":"audit_leakage.py",
               "experiment-reproducibility":"verify_reproducibility.py",
               "dashboard-qa":"verify_dashboard_qa.py"}
    for skill, script in scripts.items():
        commands.append([sys.executable, f".agents/skills/{skill}/scripts/{script}", "--self-test"])
    for mode in ["demo", "m5", "m5_intermittent"]:
        cfg = load_config(f"configs/{mode}.yaml")
        commands += [
            [sys.executable, ".agents/skills/data-leakage-audit/scripts/audit_leakage.py", "--manifest", cfg.data.processed_dir+"/lineage_manifest.json", "--scan-dir", "src"],
            [sys.executable, ".agents/skills/experiment-reproducibility/scripts/verify_reproducibility.py", "--manifest", cfg.data.artifacts_dir+"/run_manifest.json"],
            [sys.executable, ".agents/skills/dashboard-qa/scripts/verify_dashboard_qa.py", "--bundle", cfg.data.artifacts_dir+"/dashboard_qa_bundle.json"],
        ]
    rows = []
    for command in commands:
        result = subprocess.run(command,capture_output=True,text=True)
        rows.append(dict(command=command,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
    Path("reports/release/skill_checks.json").write_text(json.dumps(rows,indent=2))
    print(json.dumps(dict(checks=len(rows),failed=[row for row in rows if row["exit_code"]]),indent=2))
    return 1 if any(row["exit_code"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
