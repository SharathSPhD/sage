"""Print the gate units a changeset can actually affect.

The Gates workflow used to re-run all 26 units on every push. A unit's verdict can
only change if one of the files it names changed, so on a typical push the correct
answer is "none" and the whole sweep is dead weight. This maps changed files to the
units that reference them; the full sweep still runs nightly and on tags, which is
what catches drift that a path map cannot see.

Usage:  python gates/affected_units.py <changed-file> [...]
Prints one unit id per line. Prints nothing when no unit is affected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
GATES = REPO / "gates"

# A change to shared machinery can move any unit's numbers, so these force a full
# sweep rather than a targeted one. Deliberately broad: a false full-run costs time,
# a false skip costs a silent regression, and this project has been bitten by the
# latter (F-0018 — 23 gates green while every solver call in the shipped wheel failed).
GLOBAL_TRIGGERS = (
    "packages/strataq/strataq/core/",
    "packages/strataq/strataq/finite/",
    "packages/strataq/strataq/population/",
    "packages/strataq/strataq/thermo/",
    "packages/strataq/pyproject.toml",
    "packages/strataq-bench/",
    "pyproject.toml",
    "uv.lock",
    "gates/run_gates.py",
    "config/",
)


def _paths_in(node: object, out: set[str]) -> None:
    if isinstance(node, str):
        if "/" in node and not node.startswith("http"):
            out.add(node)
    elif isinstance(node, dict):
        for v in node.values():
            _paths_in(v, out)
    elif isinstance(node, list):
        for v in node:
            _paths_in(v, out)


def main(changed: list[str]) -> int:
    if not changed:
        return 0
    if any(c.startswith(t) for c in changed for t in GLOBAL_TRIGGERS):
        for f in sorted(GATES.glob("*.yaml")):
            if f.name != "schema.yaml":
                print(yaml.safe_load(f.read_text()).get("unit", f.stem))
        return 0
    for f in sorted(GATES.glob("*.yaml")):
        if f.name == "schema.yaml":
            continue
        spec = yaml.safe_load(f.read_text()) or {}
        declared: set[str] = set()
        _paths_in(spec.get("gates", {}), declared)
        if any(c == d or c.startswith(d.rstrip("/") + "/") for c in changed for d in declared):
            print(spec.get("unit", f.stem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
