#!/usr/bin/env python3
"""The deterministic gate runner — the only judge of gate state.

Template Method: one section class per gate section; ``run_unit`` walks them in
a fixed order and writes ``gates/status.json``. Agents dispatch on the failure
classes it reports (see .claude/skills/gate-runner/SKILL.md); nothing marks a
gate green except this script observing the checks pass.

Usage:
    python gates/run_gates.py <unit-id> [...]   # run specific units
    python gates/run_gates.py --all             # run every gates/*.yaml
    python gates/run_gates.py --check           # CI mode: run all, exit 1 if any
                                                # unit green in status.json now fails
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
GATES_DIR = REPO / "gates"
STATUS_PATH = GATES_DIR / "status.json"
STUB_PATTERN = re.compile(r"\bTODO\b|\bNotImplementedError\b|pass\s+#\s*implement", re.IGNORECASE)


@dataclass
class SectionResult:
    name: str
    failures: list[str] = field(default_factory=list)

    @property
    def green(self) -> bool:
        return not self.failures


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, check=False)
    return proc.returncode, (proc.stdout + proc.stderr)


class Section:
    """Template method base: subclasses implement ``check``."""

    name = "base"

    def __init__(self, spec: dict[str, Any], unit: str) -> None:
        self.spec = spec
        self.unit = unit

    def run(self) -> SectionResult:
        result = SectionResult(self.name)
        if not self.spec:
            return result  # absent section = nothing demanded = green
        self.check(result)
        return result

    def check(self, result: SectionResult) -> None:  # pragma: no cover - abstract
        raise RuntimeError("Section subclasses implement check().")


class CodeSection(Section):
    name = "code"

    def check(self, result: SectionResult) -> None:
        paths = [str(REPO / p) for p in self.spec.get("paths", [])]
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            result.failures.append(f"unit paths missing: {missing}")
            return
        test_paths = [p for p in paths if "/tests/" in p or "test_" in Path(p).name]
        if self.spec.get("tests_pass") and test_paths:
            rc, out = _run(["uv", "run", "pytest", "-q", *test_paths])
            if rc != 0:
                result.failures.append(f"tests_pass: pytest failed\n{out[-2000:]}")
        source_paths = [p for p in paths if p not in test_paths]
        if self.spec.get("lint_clean") and paths:
            rc, out = _run(["uv", "run", "ruff", "check", *paths])
            if rc != 0:
                result.failures.append(f"lint_clean: {out[-1000:]}")
        if self.spec.get("types_clean") and source_paths:
            rc, out = _run(["uv", "run", "mypy", *source_paths])
            if rc != 0:
                result.failures.append(f"types_clean: {out[-1000:]}")
        if self.spec.get("coverage_min") and source_paths and test_paths:
            floor = float(self.spec["coverage_min"]) * 100
            # Measure exactly the unit's own source files — never parent
            # packages, or the gate would regress whenever *sibling* modules
            # are added (this happened: core/dynamics diluted core/).
            unit_files = [p for p in source_paths if p.endswith(".py")]
            rc, out = _run(["uv", "run", "coverage", "run", "-m", "pytest", "-q", *test_paths])
            if rc != 0:
                result.failures.append(f"coverage_min: test run failed\n{out[-800:]}")
            else:
                rc, out = _run(
                    [
                        "uv",
                        "run",
                        "coverage",
                        "report",
                        f"--include={','.join(unit_files)}",
                        f"--fail-under={floor}",
                    ]
                )
                if rc != 0:
                    result.failures.append(f"coverage_min {floor:.0f}%: not met\n{out[-800:]}")
        if self.spec.get("no_todo_no_stub"):
            stubbed = [
                p
                for p in source_paths
                if Path(p).suffix == ".py" and STUB_PATTERN.search(Path(p).read_text())
            ]
            if stubbed:
                result.failures.append(f"no_todo_no_stub: stubs/TODOs in {stubbed}")


class DomainSection(Section):
    name = "domain"

    def check(self, result: SectionResult) -> None:
        for gate in self.spec if isinstance(self.spec, list) else []:
            gate_id = gate.get("id", "?")
            artifact = gate.get("artifact")
            if not artifact:
                result.failures.append(f"{gate_id}: no artifact declared")
                continue
            path = REPO / artifact
            if not path.exists():
                result.failures.append(f"{gate_id}: artifact missing ({artifact})")
                continue
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                result.failures.append(f"{gate_id}: artifact is not valid JSON")
                continue
            if data.get("passed") is not True:
                result.failures.append(f"{gate_id}: artifact reports passed={data.get('passed')}")
            if data.get("seed") is None:
                result.failures.append(
                    f"{gate_id}: artifact has no recorded seed (not regenerable)"
                )


class StatisticalSection(Section):
    name = "statistical"

    def __init__(self, spec: dict[str, Any], unit: str, domain_spec: Any) -> None:
        super().__init__(spec, unit)
        self.domain_spec = domain_spec

    def _artifacts(self) -> list[dict[str, Any]]:
        out = []
        for gate in self.domain_spec if isinstance(self.domain_spec, list) else []:
            path = REPO / gate.get("artifact", "")
            if path.exists():
                try:
                    out.append(json.loads(path.read_text()))
                except json.JSONDecodeError:
                    continue
        return out

    def check(self, result: SectionResult) -> None:
        artifacts = self._artifacts()
        stat_artifacts = [a for a in artifacts if a.get("kind") == "statistical"]
        if self.spec.get("effect_sizes_reported") and any(
            not a.get("effect_sizes") for a in stat_artifacts
        ):
            result.failures.append(
                "effect_sizes_reported: statistical artifact without effect sizes"
            )
        if self.spec.get("confidence_intervals"):
            for a in stat_artifacts:
                for es in a.get("effect_sizes", []):
                    if es.get("ci_low") is None or es.get("ci_high") is None:
                        result.failures.append(
                            f"confidence_intervals: missing in {a.get('benchmark_id')}"
                        )
        if self.spec.get("n_justified") and any(
            not a.get("n_justification") for a in stat_artifacts
        ):
            result.failures.append("n_justified: statistical artifact without n_justification")
        if self.spec.get("seeds_fixed_and_recorded") and any(
            a.get("seed") is None for a in artifacts
        ):
            result.failures.append("seeds_fixed_and_recorded: artifact without seed")


class DocumentationSection(Section):
    name = "documentation"

    def check(self, result: SectionResult) -> None:
        for doc in self.spec.get("files", []):
            if not (REPO / doc).exists():
                result.failures.append(f"docs file missing: {doc}")
        if self.spec.get("claims_ledger_updated"):
            claims_path = REPO / "memory" / "claims.md"
            claims = claims_path.read_text() if claims_path.exists() else ""
            if self.unit not in claims:
                result.failures.append(f"claims_ledger_updated: no mention of unit '{self.unit}'")
        if self.spec.get("changelog_entry"):
            changelog_path = REPO / "CHANGELOG.md"
            changelog = changelog_path.read_text() if changelog_path.exists() else ""
            if self.unit not in changelog:
                result.failures.append(f"changelog_entry: no entry referencing '{self.unit}'")


class AdversarialSection(Section):
    name = "adversarial"

    def check(self, result: SectionResult) -> None:
        if not self.spec.get("red_team_signoff"):
            result.failures.append("red_team_signoff: not granted")
        open_objections = [
            o
            for o in self.spec.get("objections", [])
            if o.get("disposition") not in ("addressed", "accepted")
        ]
        if open_objections:
            result.failures.append(f"objections open: {[o.get('id') for o in open_objections]}")


def run_unit(unit: str) -> dict[str, Any]:
    spec_path = GATES_DIR / f"{unit}.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    if spec.get("tier") == "conjectured" and not spec.get("falsifier"):
        return {
            "unit": unit,
            "green": False,
            "sections": {"meta": ["conjectured tier requires a 'falsifier' field"]},
        }
    gates = spec.get("gates", {})
    sections = [
        CodeSection(gates.get("code", {}), unit),
        DomainSection(gates.get("domain", []), unit),
        StatisticalSection(gates.get("statistical", {}), unit, gates.get("domain", [])),
        DocumentationSection(gates.get("documentation", {}), unit),
        AdversarialSection(gates.get("adversarial", {}), unit),
    ]
    results = [s.run() for s in sections]
    return {
        "unit": unit,
        "claim": spec.get("claim", ""),
        "tier": spec.get("tier", ""),
        "green": all(r.green for r in results),
        "sections": {r.name: r.failures for r in results},
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def all_units() -> list[str]:
    return sorted(p.stem for p in GATES_DIR.glob("*.yaml") if p.name not in ("schema.yaml",))


def main(argv: list[str]) -> int:
    check_mode = "--check" in argv
    units = (
        all_units()
        if ("--all" in argv or check_mode)
        else [a for a in argv if not a.startswith("-")]
    )
    if not units:
        print("usage: run_gates.py <unit-id>... | --all | --check", file=sys.stderr)
        return 64

    previous: dict[str, Any] = {}
    if STATUS_PATH.exists():
        previous = json.loads(STATUS_PATH.read_text()).get("units", {})

    results = {unit: run_unit(unit) for unit in units}
    merged = {**previous, **results}
    STATUS_PATH.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "units": merged,
            },
            indent=2,
        )
        + "\n"
    )

    exit_code = 0
    for unit, res in results.items():
        mark = "GREEN" if res["green"] else "RED"
        print(f"[{mark}] {unit}")
        for section, failures in res["sections"].items():
            for failure in failures:
                print(f"    {section}: {failure.splitlines()[0]}")
        if not res["green"]:
            exit_code = 1
    if check_mode:
        regressed = [
            u for u, prev in previous.items() if prev.get("green") and not merged[u].get("green")
        ]
        if regressed:
            print(f"REGRESSION: previously green units now red: {regressed}", file=sys.stderr)
            return 1
        return 0  # --check fails only on regression, not on units still in progress
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
