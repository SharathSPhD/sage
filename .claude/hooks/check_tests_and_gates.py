#!/usr/bin/env python3
"""Anti-gaming guards, pre-commit mode (reads the staged diff via git).

1. Test deletion / xfail-ing: removing ``def test_`` lines or adding
   ``xfail`` markers in tests requires SAGE_ADR_REF (a recorded decision).
   Closing a gate by deleting its failing test is a hard failure.
2. Gate regression: gates/status.json entries may not move from green
   (``passed``/``green``) to anything else without SAGE_ADR_REF.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def adr_ref_valid() -> bool:
    """SAGE_ADR_REF is honoured only if the referenced ADR actually exists (O-2)."""
    ref = os.environ.get("SAGE_ADR_REF", "")
    if not ref:
        return False
    decisions = REPO / "memory" / "decisions.md"
    return decisions.exists() and ref in decisions.read_text()


TEST_PATHSPECS = [":(glob)**/tests/**/*.py", ":(glob)**/test_*.py", ":(glob)**/*_test.py"]


def staged_diff(pathspecs: list[str]) -> str:
    return subprocess.run(
        ["git", "diff", "--cached", "--unified=0", "--", *pathspecs],
        capture_output=True,
        text=True,
        check=False,
    ).stdout


def staged_blob(path: str, side: str) -> str | None:
    """Content of `path` on HEAD (side='old') or in the index (side='new')."""
    ref = f"HEAD:{path}" if side == "old" else f":{path}"
    proc = subprocess.run(["git", "show", ref], capture_output=True, text=True, check=False)
    return proc.stdout if proc.returncode == 0 else None


def check_tests() -> list[str]:
    problems = []
    diff = staged_diff(TEST_PATHSPECS)
    removed_tests = re.findall(r"^-\s*def (test_\w+)", diff, re.MULTILINE)
    added_back = set(re.findall(r"^\+\s*def (test_\w+)", diff, re.MULTILINE))
    genuinely_removed = [t for t in removed_tests if t not in added_back]
    if genuinely_removed:
        problems.append(f"tests removed: {', '.join(sorted(set(genuinely_removed)))}")
    if re.search(r"^\+.*\bxfail\b", diff, re.MULTILINE):
        problems.append("new xfail marker added in tests")
    return problems


GREEN = {"green", "passed", "pass", True}


def flatten_status(obj: object, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                out.update(flatten_status(value, path))
            else:
                out[path] = value
    return out


def check_gates() -> list[str]:
    old_raw = staged_blob("gates/status.json", "old")
    new_raw = staged_blob("gates/status.json", "new")
    if old_raw is None or new_raw is None:
        return []
    try:
        old = flatten_status(json.loads(old_raw))
        new = flatten_status(json.loads(new_raw))
    except json.JSONDecodeError:
        return ["gates/status.json is not valid JSON"]
    regressed = [
        key for key, val in old.items() if val in GREEN and key in new and new[key] not in GREEN
    ]
    return [f"gate regressed: {key}" for key in regressed]


def main() -> int:
    problems = check_tests() + check_gates()
    if problems and not adr_ref_valid():
        print(
            "BLOCKED (anti-gaming guard):\n  - "
            + "\n  - ".join(problems)
            + "\nDeleting/xfail-ing failing tests or regressing a green gate requires "
            "a recorded decision: write the ADR in memory/decisions.md and set "
            "SAGE_ADR_REF=<adr-id> for this commit. The referenced ADR must exist "
            "in memory/decisions.md or the override is ignored.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
