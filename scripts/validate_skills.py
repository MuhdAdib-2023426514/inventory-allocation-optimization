#!/usr/bin/env python3
"""Validate all 5 project-specific Codex Skills under .agents/skills/.

Verifies directory layout, YAML frontmatter (name, description), required sections,
AGENTS.md reference links, relative file links, and script syntax/executability.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

EXPECTED_SKILLS = [
    "optimization-verification",
    "decision-focused-learning",
    "data-leakage-audit",
    "experiment-reproducibility",
    "dashboard-qa",
]

REQUIRED_SECTIONS = [
    "## Triggering Conditions",
    "## Required Inputs",
    "## Execution Workflow",
    "## Verification Procedures",
    "## Expected Outputs",
    "## Completion Criteria",
]


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse simple YAML frontmatter (name and multi-line or single-line description)."""
    if not text.startswith("---\n"):
        raise ValueError("File does not start with YAML frontmatter delimiter '---'")
    end_idx = text.find("\n---\n", 4)
    if end_idx == -1:
        raise ValueError("Closing YAML frontmatter delimiter '---' not found")

    raw_fm = text[4:end_idx]
    body = text[end_idx + 5 :]

    data: dict[str, str] = {}
    lines = raw_fm.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.strip().startswith("#"):
            i += 1
            continue
        if ":" not in line:
            raise ValueError(f"Malformed frontmatter line: {line!r}")
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        if val in (">-", ">", "|", "|-"):
            block_lines: list[str] = []
            i += 1
            while i < len(lines) and (lines[i].startswith("  ") or not lines[i].strip()):
                block_lines.append(lines[i].strip())
                i += 1
            data[key] = " ".join(part for part in block_lines if part)
        else:
            data[key] = val.strip("\"'")
            i += 1

    return data, body


def validate_skill_dir(skill_dir: Path) -> dict[str, Any]:
    """Validate a single skill folder under .agents/skills/<skill_name>."""
    errors: list[str] = []
    skill_name = skill_dir.name
    skill_md = skill_dir / "SKILL.md"

    if not skill_md.is_file():
        return {"skill": skill_name, "status": "FAIL", "errors": [f"Missing {skill_md}"]}

    content = skill_md.read_text(encoding="utf-8")
    try:
        fm, body = parse_frontmatter(content)
    except ValueError as exc:
        return {"skill": skill_name, "status": "FAIL", "errors": [str(exc)]}

    if fm.get("name") != skill_name:
        errors.append(f"Frontmatter name {fm.get('name')!r} != directory name {skill_name!r}")

    desc = fm.get("description", "")
    if len(desc) < 80:
        errors.append(f"Description too short ({len(desc)} chars); must be specific for auto-invocation")

    for section in REQUIRED_SECTIONS:
        if section not in body:
            errors.append(f"Missing required section {section!r} in {skill_md}")

    if "AGENTS.md" not in body:
        errors.append(f"{skill_md} must reference AGENTS.md")

    # Validate relative markdown links [label](relative/path)
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for match in link_pattern.finditer(body):
        target = match.group(1).split("#", 1)[0].strip()
        if not target or target.startswith(("http://", "https://", "mailto:", "conversation://")):
            continue
        resolved = (skill_dir / target).resolve()
        if not resolved.exists():
            errors.append(f"Broken relative link in {skill_md}: {target!r} -> {resolved}")

    scripts_dir = skill_dir / "scripts"
    refs_dir = skill_dir / "references"
    if not scripts_dir.is_dir() or not list(scripts_dir.glob("*.py")):
        errors.append(f"Skill {skill_name} must include at least one Python script in scripts/")
    if not refs_dir.is_dir() or not list(refs_dir.glob("*.md")):
        errors.append(f"Skill {skill_name} must include at least one reference doc in references/")

    return {
        "skill": skill_name,
        "status": "PASS" if not errors else "FAIL",
        "name": fm.get("name"),
        "description_length": len(desc),
        "scripts": sorted(str(p.relative_to(skill_dir)) for p in scripts_dir.glob("*.py"))
        if scripts_dir.is_dir()
        else [],
        "references": sorted(str(p.relative_to(skill_dir)) for p in refs_dir.glob("*.md"))
        if refs_dir.is_dir()
        else [],
        "errors": errors,
    }


def validate_all_skills(repo_root: Path) -> dict[str, Any]:
    """Validate all 5 expected skills under .agents/skills/."""
    skills_root = repo_root / ".agents" / "skills"
    results: list[dict[str, Any]] = []
    all_errors: list[str] = []

    for name in EXPECTED_SKILLS:
        res = validate_skill_dir(skills_root / name)
        results.append(res)
        for err in res["errors"]:
            all_errors.append(f"[{name}] {err}")

    return {
        "status": "PASS" if not all_errors else "FAIL",
        "skills_checked": len(results),
        "results": results,
        "errors": all_errors,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    report = validate_all_skills(repo_root)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
