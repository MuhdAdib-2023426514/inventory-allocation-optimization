"""Automated test suite for project-specific Codex Skills and verification scripts."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / ".agents" / "skills"


def run_script(script_rel_path: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Execute a python script with the current interpreter and capture output."""
    script_path = REPO_ROOT / script_rel_path
    return subprocess.run(
        [sys.executable, str(script_path), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


class TestCodexSkillsAndOracles(unittest.TestCase):
    """Verify structure, frontmatter, and representative tasks for all 5 Codex Skills."""

    def test_01_validate_all_skills_structure_and_frontmatter(self) -> None:
        proc = run_script("scripts/validate_skills.py")
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["skills_checked"], 5)

    def test_02_optimization_verification_skill_representative_tasks(self) -> None:
        script = ".agents/skills/optimization-verification/scripts/verify_optimization.py"
        # 1. Self-test (toy oracle, 25 random marginal cost identities, CVaR LP==reweighting)
        proc = run_script(script, "--self-test")
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)

        # 2. Representative task: validate optimal 2-store toy solution via --input
        with tempfile.TemporaryDirectory() as tmpdir:
            valid_json = Path(tmpdir) / "valid_opt.json"
            valid_json.write_text(
                json.dumps(
                    {
                        "D": [[60], [20]],
                        "I": [[0], [0]],
                        "W": [60],
                        "K": [[60], [60]],
                        "C": [60, 60],
                        "b": [[1.0], [1.0]],
                        "h": [[0.0], [0.0]],
                        "t": [[0.0], [0.0]],
                        "x": [[40], [20]],
                        "reported_cost": 20.0,
                        "oracle_cost": 20.0,
                        "solver_status": "optimal",
                    }
                ),
                encoding="utf-8",
            )
            res_ok = run_script(script, "--input", str(valid_json))
            self.assertEqual(res_ok.returncode, 0, msg=res_ok.stdout)

            # Invalid solution: violates store capacity and reports wrong cost
            invalid_json = Path(tmpdir) / "invalid_opt.json"
            invalid_json.write_text(
                json.dumps(
                    {
                        "D": [[60], [20]],
                        "I": [[10], [0]],
                        "W": [60],
                        "K": [[60], [60]],
                        "C": [45, 60],
                        "b": [[1.0], [1.0]],
                        "h": [[0.0], [0.0]],
                        "t": [[0.0], [0.0]],
                        "x": [[40], [20]],
                        "reported_cost": 999.0,
                        "oracle_cost": 10.0,
                        "solver_status": "optimal",
                    }
                ),
                encoding="utf-8",
            )
            res_bad = run_script(script, "--input", str(invalid_json))
            self.assertNotEqual(res_bad.returncode, 0)
            bad_report = json.loads(res_bad.stdout)
            self.assertEqual(bad_report["status"], "FAIL")

    def test_03_decision_focused_learning_skill_representative_tasks(self) -> None:
        script = ".agents/skills/decision-focused-learning/scripts/verify_dfl_contract.py"
        proc = run_script(script, "--self-test")
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)

        with tempfile.TemporaryDirectory() as tmpdir:
            valid_dfl = Path(tmpdir) / "valid_dfl.json"
            valid_dfl.write_text(
                json.dumps(
                    {
                        "H": 6,
                        "K_s": [3, 3],
                        "I_s": [1, 0],
                        "W_p_instances": [3, 3],
                        "b_s": [5.0, 4.0],
                        "h_s": [0.5, 0.5],
                        "t_s": [0.1, 0.1],
                        "pmf": [
                            [0.10, 0.20, 0.25, 0.20, 0.10, 0.05, 0.05, 0.05],
                            [0.15, 0.25, 0.20, 0.15, 0.10, 0.05, 0.05, 0.05],
                        ],
                        "d_true": [3, 2],
                        "oracle_splits_used": ["train", "val"],
                        "grad_norm_nll_only": 0.52,
                        "grad_norm_dfl": 1.14,
                        "param_diff_l2_vs_nll": 0.031,
                    }
                ),
                encoding="utf-8",
            )
            res_ok = run_script(script, "--input", str(valid_dfl))
            self.assertEqual(res_ok.returncode, 0, msg=res_ok.stdout)

    def test_04_data_leakage_audit_skill_representative_tasks(self) -> None:
        script = ".agents/skills/data-leakage-audit/scripts/audit_leakage.py"
        proc = run_script(script, "--self-test")
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)

        # Also test AST scan on src/
        res_scan = run_script(script, "--scan-dir", "src")
        self.assertEqual(res_scan.returncode, 0, msg=res_scan.stdout)

    def test_05_experiment_reproducibility_skill_representative_tasks(self) -> None:
        script = ".agents/skills/experiment-reproducibility/scripts/verify_reproducibility.py"
        proc = run_script(script, "--self-test")
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)

        env_proc = run_script(script, "--check-env")
        self.assertEqual(env_proc.returncode, 0, msg=env_proc.stdout)
        env_data = json.loads(env_proc.stdout)
        self.assertIn("python_version", env_data["environment"])

    def test_06_dashboard_qa_skill_representative_tasks(self) -> None:
        script = ".agents/skills/dashboard-qa/scripts/verify_dashboard_qa.py"
        proc = run_script(script, "--self-test", "--emit-playwright-plan")
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["playwright_mcp_plan"]), 9)


if __name__ == "__main__":
    unittest.main()
