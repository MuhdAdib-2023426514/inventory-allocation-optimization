"""Write a readiness report and local bundle only after actual release gates pass."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

from inventory_di.config import load_config
from inventory_di.io import sha256_file, source_provenance


def main():
    root = Path("reports/release")
    def read(name):
        return json.loads((root/name).read_text())
    independent = read("independent_verification.json")
    assert len(independent) == 3 and all(r["status"] == "PASS" for r in independent.values())
    pipeline = read("pipeline_execution.json")
    assert len(pipeline) == 12 and all(r["exit_code"] == 0 for r in pipeline)
    assert len(read("clean_pipeline.json")) == 4 and all(r["exit_code"] == 0 for r in read("clean_pipeline.json"))
    reproducibility = read("clean_reproducibility.json")
    assert reproducibility["costs_allocations_deterministic_fields_match"] and all(reproducibility["checkpoint_hash_match"].values())
    browser = read("browser/qa_demo_m5_m5_intermittent.json")
    assert browser["status"] == "PASS" and len(browser["pages"]) == 15 and browser["slides_fit"]
    assert len(browser["downloads"]) == 3 and not browser["external_attempts"]
    offline = read("offline_network_probe.json")
    assert offline["loopback"]["success"] and not offline["external_ip"]["success"]
    app = read("app_qa_demo_m5_m5_intermittent.json")
    assert app["status"] == "PASS" and app["count"] >= 45
    presentation = read("presentation_verification.json")
    assert presentation["status"] == "PASS" and len(presentation["slides"]) == 7
    skills = read("skill_checks.json")
    assert all(row["exit_code"] == 0 for row in skills)
    pytest = (root/"pytest.log").read_text()
    assert not re.search(r"\d+ (?:failed|xfailed|skipped)",pytest)
    test_count = int(re.search(r"(\d+) passed",pytest).group(1))
    assert "All checks passed" in (root/"ruff.log").read_text()
    content = read("presentation_content.json")
    assert all(sha256_file(p) == digest for p,digest in content["sources"].items())
    checked = sum(r["checked_policy_origins"] for r in independent.values())
    closure = {
        "R1": "Current-stock marginal costs; rolling regression; new v2 carryover runs",
        "R2": "Dispatch/admission/deferred oracle; independent enumeration; replay and nonnegative regret checks",
        "R3": "Artifact-derived seven-slide decks, notes and measured results; unsupported claims removed",
        "R4": "Raw SPO+ gradients, matched update check and recomputed calibrated-checkpoint validation regret",
        "R5": "Matched joint learning, equal seed budgets and seed-specific validation/risk selection",
        "R6": "Hard/soft service modes, separate maximum-fill solve and brute-force service verification",
        "R7": "Common coherent tail/calibrated marginals, randomized ties and risk-neutral scenario ablations",
        "R8": "Zero supply retained; regression tests",
        "R9": "Historical-only prices, unique validated joins and stricter finite/integer/temporal contracts",
        "R10": "Effective configs, source/input/output lineage, profile isolation, accepted solver statuses and objective checks",
        "R11": "Cache invalidation, validation defaults, complete CSV metrics and offline browser/AppTest checks",
        "R12": "Locked presentation/QA groups, installed-package doctor and clean pipeline, actual PowerPoint/HTML layout checks",
    }
    lines = ["# Interview readiness report — corrected v2 release", "",
        "**Technical verdict: ready for an interview research-prototype presentation in the verified local environment.**",
        "This does not establish production readiness, actual retailer savings, or universal model superiority.", "",
        f"- Automated suite: **{test_count} passed**, no known-defect expected failures; Ruff passed.",
        f"- Independent numerical checks: **{checked} policy/origin instances** across three modes; costs, feasibility, summaries, hashes and regret checks passed.",
        "- Full synthetic, regular-M5 and intermittent-M5 pipelines completed, with equal-budget independent seeds and validation selection.",
        "- A clean installed package completed the synthetic pipeline outside the checkout; quick-run checkpoint hashes and decision metrics reproduced exactly (runtime excluded).",
        "- Browser: all 15 page/mode combinations, risk controls and CSV downloads passed in an isolated browser with external HTTP blocked; no console or page errors.",
        "- Offline: dashboard server ran under the recorded OS loopback-only policy; local health succeeded and an external IP connection was denied.",
        "- Presentation: seven slides exported through local Microsoft PowerPoint; complete expected text, canvas bounds and footer separation checked; HTML slides checked and PDF fallback exported.", "",
        "## Scope and limitations", "",
        "Results are retrospective on previously inspected test periods, with M5 sales proxies, synthetic MU economics, small selected cohorts, short test horizons and three seeds. Primary result tables identify the predeclared seed; seed variation is retained in reports. No statistical or operational guarantee is implied.",
        "Candidate delivery rehearsal is prepared, not claimed performed. Personally rehearse the nine-minute talk and sixty-second dashboard path before the interview. The software checks cannot certify speaking delivery or an untested future laptop setup.", "",
        "## Finding closure", "", "| Finding | Implementation and evidence |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key,value in closure.items()]
    lines += ["", "## Evidence", "",
        "See `reports/release/`: pipeline/clean execution logs, independent verification, skill checks, browser QA/screenshots, offline probe, presentation content/source hashes and PowerPoint verification. The release manifest identifies the exact artifact files.",
        "", "Use `PRESENTATION_GUIDE.md`, `PRESENTATION_SPEAKER_NOTES.md` and `OFFLINE_DEMO_GUIDE.md`. Historical v1 presentations are archived and must not be used as current evidence.", ""]
    Path("docs/INTERVIEW_READINESS_REPORT.md").write_text("\n".join(lines))
    Path("docs/IMPLEMENTATION_STATUS.md").write_text("# Implementation status\n\nCorrected v2 research-prototype release: technical gates passed. See [INTERVIEW_READINESS_REPORT.md](INTERVIEW_READINESS_REPORT.md) for evidence, scope and remaining scientific limitations.\n\nAll R1–R12 findings are addressed and corrected experiments have been regenerated. Candidate delivery rehearsal remains a personal handoff step, not an automated verification claim. Historical v1 outputs remain separate.\n")
    plan = Path("docs/INTERVIEW_READINESS_IMPLEMENTATION_PLAN.md")
    text = plan.read_text().replace("- [ ] Phase", "- [x] Phase")
    text = re.sub(r"\*\*Status:\*\*[^\n]+", "**Status:** Executed; technical release gates passed. Candidate personal rehearsal is prepared and remains a handoff step.", text)
    plan.write_text(text)
    chosen = set()
    for folder in ["src", "app", "scripts", "tests", ".agents/skills"]:
        chosen.update(p for p in Path(folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    chosen.update(Path(p) for p in ["README.md","AGENTS.md","pyproject.toml","uv.lock",".gitignore"])
    chosen.update(p for p in Path("docs").glob("*") if p.is_file())
    chosen.update(Path("configs").glob("*.yaml"))
    for mode in ["demo","m5","m5_intermittent"]:
        cfg = load_config(f"configs/{mode}.yaml")
        for folder in [cfg.data.processed_dir, cfg.data.artifacts_dir, cfg.data.reports_dir]:
            chosen.update(p for p in Path(folder).rglob("*") if p.is_file())
    chosen.update(p for p in root.rglob("*") if p.is_file() and p.suffix not in {".zip"})
    chosen = {p for p in chosen if p.name not in {"release_manifest.json","bundle_sha256.txt"}}
    manifest = dict(status="research_prototype_presentation_ready", date="2026-10-29",
        source=source_provenance(), run_ids=content["run_ids"], closure=closure,
        candidate_rehearsal="Prepared; personal rehearsal not claimed", verified_policy_origins=checked,
        tests_passed=test_count, files={str(p):sha256_file(p) for p in sorted(chosen)})
    path = root/"release_manifest.json"
    path.write_text(json.dumps(manifest,indent=2))
    chosen.add(path)
    bundle = Path("reports/interview_release_v2.zip")
    with zipfile.ZipFile(bundle,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(chosen):
            assert not str(p).startswith("data/raw/")
            archive.write(p,str(p))
    (root/"bundle_sha256.txt").write_text(sha256_file(bundle)+"  "+str(bundle)+"\n")
    print(json.dumps(dict(status=manifest["status"], policy_origins=checked, tests_passed=test_count,
                         bundle=str(bundle), bytes=bundle.stat().st_size),indent=2))


if __name__ == "__main__":
    main()
