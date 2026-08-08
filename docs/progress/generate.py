#!/usr/bin/env python3
"""Generate the progress dashboard (docs/progress/index.md) from live project state.

Inputs: gates/status.json, benchmarks/results/*.json, memory/claims.md,
memory/findings.md. Regenerated on every merge to main (docs workflow) and by
`make dashboard`. A stale dashboard means the gate is not green.

Two synchronised views: technical (gate matrix, claim tiers, benchmark state,
open objections, anomaly log) and non-technical (plain language, the α axis,
track health). Greek letters in the non-technical view carry hover definitions
from memory/glossary.md.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "index.md"

SECTION_ORDER = ["code", "domain", "statistical", "documentation", "adversarial"]

# The α axis: (position 0..1, label, anchor state key)
ALPHA_DOMAINS = [
    (0.0, "congestion", "population engine"),
    (0.35, "pricing", "finite engine"),
    (0.5, "electricity", "finite engine"),
    (0.9, "blotto", "finite engine"),
    (1.0, "RPS", "test family"),
]

TRACKS = {
    "A · Engine 1 core": ["finite.", "core."],
    "B · Calibration": ["population.", "domains.congestion", "domains.blotto"],
    "C · Empirics": ["domains.pricing", "domains.electricity", "estimate."],
    "D · Product": ["api.", "web."],
    "Foundation": ["stage0"],
}


def load_status() -> dict:
    path = REPO / "gates" / "status.json"
    if not path.exists():
        return {"units": {}}
    return json.loads(path.read_text())


def load_bench() -> list[dict]:
    results = []
    for path in sorted((REPO / "benchmarks" / "results").glob("*.json")):
        try:
            results.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return results


def glossary_defs() -> dict[str, str]:
    """symbol -> one-line definition, parsed from memory/glossary.md bullets."""
    defs: dict[str, str] = {}
    gpath = REPO / "memory" / "glossary.md"
    if not gpath.exists():
        return defs
    for line in gpath.read_text().splitlines():
        m = re.match(r"- \*\*(.+?)\*\* — (.+)", line)
        if m:
            term = m.group(1).split(" ")[0]
            defs[term] = m.group(2).split(".")[0]
    return defs


def hover(symbol: str, defs: dict[str, str]) -> str:
    definition = defs.get(symbol, "")
    return f'<abbr title="{definition}">{symbol}</abbr>' if definition else symbol


def claims_by_tier() -> dict[str, int]:
    counts = {"exact": 0, "derived": 0, "conjectured": 0, "speculative": 0}
    cpath = REPO / "memory" / "claims.md"
    if not cpath.exists():
        return counts
    text = cpath.read_text()
    counts["exact"] = len(re.findall(r"^\| K\d+", text, re.MULTILINE))
    counts["derived"] = len(re.findall(r"^\| (R\d+|N\d+)", text, re.MULTILINE))
    counts["conjectured"] = len(re.findall(r"^\| C\d+", text, re.MULTILINE))
    counts["speculative"] = len(re.findall(r"^\| S\d+", text, re.MULTILINE))
    return counts


def findings_count() -> int:
    fpath = REPO / "memory" / "findings.md"
    if not fpath.exists():
        return 0
    return len(re.findall(r"^## F-\d+", fpath.read_text(), re.MULTILINE))


def alpha_axis_svg(units: dict) -> str:
    """The α axis with domains placed and anchors marked green when their gate is green."""
    width, height, pad = 700, 110, 40
    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="alpha axis with domain anchors">',
        f'<line x1="{pad}" y1="60" x2="{width - pad}" y2="60" stroke="currentColor" stroke-width="2"/>',
        f'<text x="{pad}" y="95" font-size="13" fill="currentColor">α = 0 (potential — everything known)</text>',
        f'<text x="{width - pad}" y="95" font-size="13" text-anchor="end" fill="currentColor">α = 1 (harmonic — cycles live here)</text>',
    ]
    for pos, name, _engine in ALPHA_DOMAINS:
        x = pad + pos * (width - 2 * pad)
        anchored = any(
            unit_id.startswith(f"domains.{name}") and unit.get("green")
            for unit_id, unit in units.items()
        )
        fill = "#2e7d32" if anchored else "#9e9e9e"
        parts.append(f'<circle cx="{x:.0f}" cy="60" r="7" fill="{fill}"/>')
        parts.append(
            f'<text x="{x:.0f}" y="40" font-size="12" text-anchor="middle" fill="currentColor">{name}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def gate_matrix(units: dict) -> str:
    if not units:
        return "_No gates recorded yet._"
    header = "| Unit | " + " | ".join(SECTION_ORDER) + " | Overall |\n"
    header += "|---|" + "---|" * (len(SECTION_ORDER) + 1) + "\n"
    rows = []
    for unit_id in sorted(units):
        unit = units[unit_id]
        cells = []
        for section in SECTION_ORDER:
            failures = unit.get("sections", {}).get(section)
            cells.append("🟢" if not failures else "🔴")
        overall = "🟢" if unit.get("green") else "🔴"
        rows.append(f"| `{unit_id}` | " + " | ".join(cells) + f" | {overall} |")
    return header + "\n".join(rows)


def track_health(units: dict) -> str:
    rows = ["| Track | State |", "|---|---|"]
    for track, prefixes in TRACKS.items():
        track_units = {
            uid: u for uid, u in units.items() if any(uid.startswith(p) for p in prefixes)
        }
        if not track_units:
            state = "not started"
        elif all(u.get("green") for u in track_units.values()):
            state = f"healthy — {len(track_units)} unit(s) closed"
        else:
            open_units = [uid for uid, u in track_units.items() if not u.get("green")]
            state = f"in progress — open: {', '.join(f'`{u}`' for u in open_units)}"
        rows.append(f"| {track} | {state} |")
    return "\n".join(rows)


def open_objections(units: dict) -> list[str]:
    out = []
    for unit_id, unit in units.items():
        for failure in unit.get("sections", {}).get("adversarial", []) or []:
            out.append(f"`{unit_id}`: {failure}")
    return out


def main() -> None:
    status = load_status()
    units = status.get("units", {})
    bench = load_bench()
    defs = glossary_defs()
    tiers = claims_by_tier()
    generated = status.get("generated_at", "never")

    n_green = sum(1 for u in units.values() if u.get("green"))
    bench_lines = (
        "\n".join(
            f"- `{b.get('benchmark_id')}` ({b.get('kind')}) — "
            f"{'passed' if b.get('passed') else 'FAILED'}, unit `{b.get('unit')}`"
            for b in bench
        )
        or "_No benchmark results yet — instruments arrive in Stage 1._"
    )
    objections = open_objections(units)
    objections_md = "\n".join(f"- {o}" for o in objections) or "_None open._"

    alpha = hover("α", defs)
    lam = hover("λ", defs)
    rdefect = hover("ℛ", defs)

    page = f"""# Progress

*Generated {datetime.now(UTC).isoformat(timespec="seconds")} from gate state of {generated}. Regenerated on every merge — if this page is stale, the gate is not green.*

=== "Plain language"

    **What this project is doing.** Building measuring instruments for strategic systems — how sharply players respond to incentives ({lam}), how far a system is from the "well-behaved" regime where everything settles down ({alpha}), and whether give-and-take between players is balanced ({rdefect}) — then pointing those instruments at road networks, pricing data, electricity markets and game experiments.

    **What works now.** {n_green} work unit(s) fully closed (every closure includes an adversarial review by a hostile reviewer who never sees the authors' reasoning). Claim ledger: {tiers["exact"]} established results implemented, {tiers["derived"]} results of our own, {tiers["conjectured"]} open conjectures each with a stated way to be proven wrong.

    **The map.** Each dot is a system we point the instruments at; green means that system's anchor is in place.

    {alpha_axis_svg(units)}

    **Track health**

{indent(track_health(units), 4)}

    **What's next.** The first new artefact in the world: the reciprocity meter reading exactly zero on a road-congestion game and clearly positive on rock-paper-scissors — the same measurement, two systems, opposite readings. Everything else follows from that working.

    **Anomalies logged:** {findings_count()} (anomalies are the product — each gets chased).

=== "Technical"

    **Gate matrix** (a unit closes on domain validation, not green tests)

{indent(gate_matrix(units), 4)}

    **Claims ledger** — {tiers["exact"]} `exact` · {tiers["derived"]} `derived` · {tiers["conjectured"]} `conjectured` · {tiers["speculative"]} `speculative` — [full ledger](https://github.com/SharathSPhD/sage/blob/main/memory/claims.md)

    **Benchmark results**

{indent(bench_lines, 4)}

    **Open red-team objections**

{indent(objections_md, 4)}

    **The (λ, α) phase map** — the money figure, filling in

    <img src="phase_map.svg" alt="phase map heat maps" style="max-width:100%"/>

    **Gate flow** — what "closed" means

    ```mermaid
    flowchart LR
        W[work unit] --> C{{code gates}}
        C -->|tests, types, lint, coverage, no stubs| D{{domain gates}}
        D -->|claim true in its domain, artifacts regenerable| S{{statistical gates}}
        S -->|effect sizes, CIs, n justified, seeds| Doc{{documentation}}
        Doc --> A{{adversarial}}
        A -->|red-team sign-off, objections dispositioned| G[merge to main + dashboard refresh]
        A -->|same failure twice| T[TRIZ escalation]
        T --> C
    ```
"""
    OUT.write_text(page)
    print(f"wrote {OUT}")


def indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in text.splitlines())


if __name__ == "__main__":
    main()
