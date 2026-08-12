#!/usr/bin/env python3
"""SAGE context graph — a queryable map of the project for humans and agents.

Why this exists
---------------
The project's knowledge is spread across 26 gate files, 45 artifacts, 18
pre-registered configs, 21 findings, a claims ledger, three generations of
research documents, a library, an API and an app. A newcomer (human or agent)
that reads them in file order learns the pieces without the connections.

This tool builds a graph whose BACKBONE IS EXTRACTED FROM THE REPO, so it
cannot drift from reality: units, claims, artifacts, configs, findings and
their cross-references are parsed from the files themselves. Conceptual edges
that no parser can infer (this theory motivates that instrument) are layered
on from a curated TSV. Re-run ``build`` after any change and the map is current.

Usage
-----
    uv run python docs/context_graph.py build            # regenerate the graph
    uv run python docs/context_graph.py stats            # what is in it
    uv run python docs/context_graph.py unit thermo.hs_estimator.gate_se
    uv run python docs/context_graph.py finding F-0020
    uv run python docs/context_graph.py search relaxation
    uv run python docs/context_graph.py neighbours concept:reciprocity-defect-R
    uv run python docs/context_graph.py path finding:F-0016 finding:F-0021
    uv run python docs/context_graph.py orphans            # nodes nothing links to

Start here if you are new: ``stats``, then ``unit`` on any GREEN unit, then
``path`` between two findings to see how the programme's reasoning chained.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
GRAPH = REPO / "docs" / "context-graph.json"
CURATED = REPO / "docs" / "context-graph-curated.tsv"

# Relations the curated layer may use. Kept closed so the graph stays legible;
# add deliberately rather than letting free-text relations accumulate.
RELATIONS = {
    # structural (extracted)
    "HAS_CLAIM",
    "PRODUCES",
    "REGISTERS",
    "EVIDENCES",
    "TESTED_BY",
    "DOCUMENTED_IN",
    "SIGNED_OFF",
    "NOT_SIGNED_OFF",
    "REFERENCES",
    # conceptual (curated)
    "DEFINES",
    "SUPERSEDES",
    "EXPLAINS",
    "MEASURES",
    "CITES",
    "DERIVES_FROM",
    "PROMISES_PANEL",
    "OPEN_QUESTION",
    "CONTAINS",
    "IMPLEMENTS",
    "DEPENDS_ON",
    "EXPOSES",
    "VALIDATED_BY",
    "ENFORCED_BY",
    "RETRACTS",
    "CORRECTS",
    "BLOCKS",
    "FOLLOWS_FROM",
    "SERVES",
    "CALLS",
    "DEPLOYS",
    "GATES",
    "PUBLISHES",
    "SURFACES",
    "ENFORCES",
    "REQUIRES",
    "INVOKED_BY",
    "VALIDATES",
    "NEXT_AFTER",
    "BLOCKED_BY",
    "OUTSTANDING",
    "DONE",
    "NEEDS_OPERATOR",
}


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def build() -> dict[str, Any]:
    """Extract the structural backbone from the repo, then merge curated edges."""
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []

    def node(nid: str, ntype: str, **attrs: Any) -> str:
        cur = nodes.setdefault(nid, {"id": nid, "type": ntype})
        cur.update({k: v for k, v in attrs.items() if v is not None})
        return nid

    def edge(src: str, rel: str, dst: str, note: str = "") -> None:
        edges.append({"src": src, "rel": rel, "dst": dst, "note": note})

    # ---- gate units (the spine) ------------------------------------------
    import yaml

    status_path = REPO / "gates" / "status.json"
    status = json.loads(_read(status_path)).get("units", {}) if status_path.exists() else {}

    seen_units: set[str] = set()
    for gf in sorted((REPO / "gates").glob("*.yaml")):
        try:
            g = yaml.safe_load(_read(gf))
        except yaml.YAMLError:
            continue
        if not isinstance(g, dict) or "unit" not in g:
            continue
        unit = g["unit"]
        # gates/schema.yaml carries an EXAMPLE unit name; the first file wins
        if unit in seen_units:
            continue
        seen_units.add(unit)
        uid = node(
            f"unit:{unit}",
            "unit",
            file=str(gf.relative_to(REPO)),
            tier=g.get("tier"),
            green=bool(status.get(unit, {}).get("green")),
            claim=(g.get("claim") or "").strip(),
        )
        gates = g.get("gates") or {}

        for spec in gates.get("domain") or []:
            if not isinstance(spec, dict):
                continue
            art = spec.get("artifact")
            if art:
                aid = node(f"artifact:{Path(art).name}", "artifact", path=art)
                edge(uid, "PRODUCES", aid, (spec.get("spec") or "")[:200])

        for p in (gates.get("code") or {}).get("paths") or []:
            kind = "test" if "/tests/" in p else "code"
            edge(
                uid,
                "TESTED_BY" if kind == "test" else "REFERENCES",
                node(f"{kind}:{p}", kind, path=p),
                "",
            )

        for p in (gates.get("documentation") or {}).get("files") or []:
            edge(uid, "DOCUMENTED_IN", node(f"doc:{p}", "doc", path=p), "")

        adv = gates.get("adversarial") or {}
        signed = bool(adv.get("red_team_signoff"))
        for obj in adv.get("objections") or []:
            if not isinstance(obj, dict):
                continue
            oid = node(
                f"objection:{unit}:{obj.get('id')}",
                "objection",
                text=(obj.get("text") or "")[:400],
                disposition=obj.get("disposition"),
            )
            edge(
                uid,
                "SIGNED_OFF" if signed else "NOT_SIGNED_OFF",
                oid,
                (obj.get("where") or "")[:300],
            )

    # ---- pre-registered configs ------------------------------------------
    for cf in sorted((REPO / "config" / "experiments").glob("*.yaml")):
        rel = str(cf.relative_to(REPO))
        text = _read(cf)
        cid = node(
            f"config:{cf.name}",
            "config",
            path=rel,
            registers_criteria=bool(re.search(r"REGISTERED CRITERIA", text)),
        )
        # a config names its unit in a comment header more often than a field
        for m in re.finditer(r"unit ([a-z0-9_.]+\.[a-z0-9_.]+)", text[:2000]):
            u = f"unit:{m.group(1)}"
            if u in nodes:
                edge(cid, "REGISTERS", u, "pre-registered criteria")
                break

    # ---- experiments: the reliable config <-> unit <-> artifact link ------
    # Each experiments/*.py declares UNIT, CONFIG and the artifact it writes.
    # Parsing those constants beats regexing config comment headers, which name
    # their unit inconsistently (that left 50 orphans in the first build).
    for xf in sorted((REPO / "experiments").glob("*.py")):
        text = _read(xf)
        if xf.name.startswith("_"):
            continue
        xid = node(f"experiment:{xf.stem}", "experiment", path=str(xf.relative_to(REPO)))
        m = re.search(r'^UNIT\s*=\s*["\']([^"\']+)["\']', text, re.M)
        if m and f"unit:{m.group(1)}" in nodes:
            edge(
                f"unit:{m.group(1)}",
                "PRODUCES",
                xid,
                "the experiment that generates this unit's artifact",
            )
        cm = re.search(r'CONFIG\s*=\s*[^\n]*?["\']([a-z0-9_]+\.yaml)["\']', text)
        if cm:
            cid = f"config:{cm.group(1)}"
            if cid in nodes:
                edge(xid, "REGISTERS", cid, "criteria this experiment runs against")
                if m and f"unit:{m.group(1)}" in nodes:
                    edge(
                        cid,
                        "REGISTERS",
                        f"unit:{m.group(1)}",
                        "pre-registered criteria for the unit",
                    )
        for am in re.finditer(r'["\']([a-z0-9_]+\.json)["\']', text):
            aid = f"artifact:{am.group(1)}"
            if aid in nodes:
                edge(xid, "PRODUCES", aid, "")

    # ---- artifacts (fill in ones no gate references) ----------------------
    for af in sorted((REPO / "benchmarks" / "results").glob("*.json")):
        raw = _read(af)
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            d = {}
        is_bench = isinstance(d, dict) and "benchmark_id" in d and "library_version" in d
        aid = node(
            f"artifact:{af.name}",
            "artifact",
            path=str(af.relative_to(REPO)),
            benchmark_result_shaped=is_bench,
            n_metrics=len(d.get("metrics", {})) if is_bench else None,
            n_effect_sizes=len(d.get("effect_sizes", [])) if is_bench else None,
        )
        if is_bench and d.get("unit"):
            u = f"unit:{d['unit']}"
            if u in nodes:
                edge(u, "PRODUCES", aid, "artifact declares this unit")

    # ---- findings ---------------------------------------------------------
    findings_md = _read(REPO / "memory" / "findings.md")
    blocks = re.split(r"^## (F-\d{4})", findings_md, flags=re.M)
    for i in range(1, len(blocks) - 1, 2):
        fid_raw, body = blocks[i], blocks[i + 1]
        title = body.splitlines()[0].strip(" —-")
        chase = re.search(r"Chase status:\s*([^\n]+)", body)
        fid = node(
            f"finding:{fid_raw}",
            "finding",
            title=title[:300],
            chase_status=(chase.group(1).strip() if chase else None),
            retraction=bool(re.search(r"RETRACT|retracted|CORRECTION", body)),
        )
        # findings name their unit and their artifacts
        for m in re.finditer(r"unit ([a-z0-9_.]+\.[a-z0-9_.]+)", body):
            u = f"unit:{m.group(1)}"
            if u in nodes:
                edge(u, "EVIDENCES", fid, "finding recorded for this unit")
        for m in re.finditer(r"`([a-z0-9_]+\.json)`", body):
            a = f"artifact:{m.group(1)}"
            if a in nodes:
                edge(fid, "REFERENCES", a, "")
        for m in re.finditer(r"\b(F-\d{4})\b", body):
            if m.group(1) != fid_raw:
                edge(fid, "FOLLOWS_FROM", f"finding:{m.group(1)}", "cross-referenced")

    # ---- ADRs -------------------------------------------------------------
    for m in re.finditer(r"^#+\s*(ADR-\d{4})[^\n]*", _read(REPO / "memory" / "decisions.md"), re.M):
        node(f"adr:{m.group(1)}", "adr", title=m.group(0).lstrip("# ").strip()[:200])

    # ---- claims ledger ----------------------------------------------------
    # Claim rows name their units and findings inline, e.g. "(unit thermo.protocols,
    # F-0012)". Linking those keeps the ledger reachable instead of leaving it as a
    # dozen orphan nodes — the ledger is the thing a reviewer checks a claim against.
    for m in re.finditer(
        r"^\|\s*([KNCP]\d+)\s*\|\s*([^|]{10,4000})", _read(REPO / "memory" / "claims.md"), re.M
    ):
        cid = node(f"claim:{m.group(1)}", "claim", text=m.group(2).strip()[:400])
        body = m.group(2)
        for um in re.finditer(r"unit ([a-z0-9_.]+\.[a-z0-9_.]+)", body):
            if f"unit:{um.group(1)}" in nodes:
                edge(f"unit:{um.group(1)}", "EVIDENCES", cid, "claim entered from this unit")
        for fm in re.finditer(r"\b(F-\d{4})\b", body):
            if f"finding:{fm.group(1)}" in nodes:
                edge(cid, "REFERENCES", f"finding:{fm.group(1)}", "")

    # ---- curated conceptual layer ----------------------------------------
    unknown_rel: set[str] = set()
    if CURATED.exists():
        for line in _read(CURATED).splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue
            src, rel, dst = parts[0], parts[1], parts[2]
            note = parts[3] if len(parts) > 3 else ""
            if rel not in RELATIONS:
                unknown_rel.add(rel)
                continue
            for nid in (src, dst):
                if nid not in nodes:
                    node(nid, nid.split(":", 1)[0] if ":" in nid else "concept", curated=True)
            edge(src, rel, dst, note)

    graph = {
        "generated_from": "docs/context_graph.py build",
        "repo": str(REPO),
        "nodes": list(nodes.values()),
        "edges": edges,
        "unknown_relations_skipped": sorted(unknown_rel),
    }
    GRAPH.write_text(json.dumps(graph, indent=1) + "\n")
    return graph


def load() -> dict[str, Any]:
    if not GRAPH.exists():
        print(f"no graph yet — run: uv run python {Path(__file__).name} build", file=sys.stderr)
        raise SystemExit(2)
    return json.loads(_read(GRAPH))


def _adj(g: dict[str, Any]) -> tuple[dict[str, list], dict[str, list]]:
    out, inn = defaultdict(list), defaultdict(list)
    for e in g["edges"]:
        out[e["src"]].append(e)
        inn[e["dst"]].append(e)
    return out, inn


def cmd_stats(g: dict[str, Any]) -> None:
    by_type: dict[str, int] = defaultdict(int)
    for n in g["nodes"]:
        by_type[n["type"]] += 1
    by_rel: dict[str, int] = defaultdict(int)
    for e in g["edges"]:
        by_rel[e["rel"]] += 1
    print(f"{len(g['nodes'])} nodes, {len(g['edges'])} edges\n")
    print("NODES BY TYPE")
    for t, c in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"  {c:5d}  {t}")
    print("\nEDGES BY RELATION")
    for r, c in sorted(by_rel.items(), key=lambda kv: -kv[1]):
        print(f"  {c:5d}  {r}")
    units = [n for n in g["nodes"] if n["type"] == "unit"]
    green = [n for n in units if n.get("green")]
    print(f"\nUNITS: {len(units)} total, {len(green)} GREEN, {len(units) - len(green)} not")
    for n in units:
        if not n.get("green"):
            print(f"  NOT GREEN: {n['id']}")
    if g.get("unknown_relations_skipped"):
        print(f"\nWARNING skipped unknown relations: {g['unknown_relations_skipped']}")


def _show(n: dict[str, Any], out: dict, inn: dict) -> None:
    print(f"\n=== {n['id']}  [{n['type']}]")
    for k, v in n.items():
        if k in ("id", "type"):
            continue
        s = str(v)
        print(f"  {k}: {s[:600]}{'…' if len(s) > 600 else ''}")
    if out[n["id"]]:
        print("\n  --> OUT")
        for e in out[n["id"]]:
            print(f"    {e['rel']:16s} {e['dst']}{('  # ' + e['note'][:120]) if e['note'] else ''}")
    if inn[n["id"]]:
        print("\n  <-- IN")
        for e in inn[n["id"]]:
            print(f"    {e['rel']:16s} {e['src']}{('  # ' + e['note'][:120]) if e['note'] else ''}")


def cmd_node(g: dict[str, Any], nid: str) -> None:
    out, inn = _adj(g)
    idx = {n["id"]: n for n in g["nodes"]}
    if nid not in idx:
        cands = [k for k in idx if nid.lower() in k.lower()]
        print(
            f"no exact node '{nid}'."
            + (" did you mean:\n  " + "\n  ".join(cands[:15]) if cands else "")
        )
        raise SystemExit(1)
    _show(idx[nid], out, inn)


def cmd_search(g: dict[str, Any], term: str) -> None:
    t = term.lower()
    hits = [
        n
        for n in g["nodes"]
        if t in n["id"].lower() or any(t in str(v).lower() for k, v in n.items() if k != "id")
    ]
    print(f"{len(hits)} node(s) matching '{term}'\n")
    for n in hits[:60]:
        extra = n.get("title") or n.get("claim") or n.get("text") or n.get("path") or ""
        print(f"  {n['id']:60s} {str(extra)[:110]}")
    if len(hits) > 60:
        print(f"  … {len(hits) - 60} more")


def cmd_path(g: dict[str, Any], a: str, b: str) -> None:
    adj = defaultdict(list)
    for e in g["edges"]:
        adj[e["src"]].append((e["dst"], e["rel"]))
        adj[e["dst"]].append((e["src"], f"~{e['rel']}"))  # undirected walk
    q, seen = deque([(a, [a], [])]), {a}
    while q:
        cur, path, rels = q.popleft()
        if cur == b:
            print(
                " -> ".join(
                    f"{p}" + (f"  [{r}]  " if i < len(rels) else "")
                    for i, (p, r) in enumerate(zip(path, [*rels, ""], strict=False))
                )
            )
            for i, r in enumerate(rels):
                print(f"  {path[i]}\n    --{r}-->\n  {path[i + 1]}")
            return
        for nxt, rel in adj[cur]:
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, [*path, nxt], [*rels, rel]))
    print(f"no path between {a} and {b}")


def cmd_orphans(g: dict[str, Any]) -> None:
    linked = {e["src"] for e in g["edges"]} | {e["dst"] for e in g["edges"]}
    orphans = [n for n in g["nodes"] if n["id"] not in linked]
    print(f"{len(orphans)} unlinked node(s) — usually means an artifact no gate claims,")
    print("or a curated node whose counterpart was renamed:\n")
    for n in orphans:
        print(f"  {n['id']:60s} {n.get('path') or ''}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("stats")
    sub.add_parser("orphans")
    for name, arg in (
        ("unit", "name"),
        ("finding", "id"),
        ("node", "id"),
        ("neighbours", "id"),
        ("search", "term"),
    ):
        sp = sub.add_parser(name)
        sp.add_argument(arg)
    sp = sub.add_parser("path")
    sp.add_argument("a")
    sp.add_argument("b")
    a = p.parse_args()

    if a.cmd == "build":
        g = build()
        print(f"built {GRAPH.relative_to(REPO)}: {len(g['nodes'])} nodes, {len(g['edges'])} edges")
        if g["unknown_relations_skipped"]:
            print(f"skipped unknown relations: {g['unknown_relations_skipped']}")
        return 0

    g = load()
    if a.cmd == "stats":
        cmd_stats(g)
    elif a.cmd == "orphans":
        cmd_orphans(g)
    elif a.cmd == "search":
        cmd_search(g, a.term)
    elif a.cmd == "path":
        cmd_path(g, a.a, a.b)
    elif a.cmd == "unit":
        cmd_node(g, a.name if a.name.startswith("unit:") else f"unit:{a.name}")
    elif a.cmd == "finding":
        cmd_node(g, a.id if a.id.startswith("finding:") else f"finding:{a.id}")
    else:
        cmd_node(g, a.id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
