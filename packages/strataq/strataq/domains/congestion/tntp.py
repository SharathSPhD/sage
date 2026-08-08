"""TNTP loader: parse the standard transportation-network test format.

Networks from ``github.com/bstabler/TransportationNetworks`` (Sioux Falls for
debug scale — the repo's own README warns it is not realistic, which is fine
for calibration). Fetched at runtime and cached; raw data is never committed.

References
----------
Stabler–Bar-Gera–Sall, TransportationNetworks repository; BPR cost form.
Engineering layer (Repository pattern).
"""

from __future__ import annotations

import heapq
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

RAW_BASE = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/master"
CACHE = Path.home() / ".cache" / "strataq" / "tntp"


@dataclass(frozen=True)
class TNTPNetwork:
    """Parsed link table + OD matrix, 1-indexed nodes as in the format."""

    name: str
    n_nodes: int
    init_node: tuple[int, ...]
    term_node: tuple[int, ...]
    capacity: tuple[float, ...]
    free_flow_time: tuple[float, ...]
    b: tuple[float, ...]
    power: tuple[float, ...]
    demand: dict[tuple[int, int], float] = field(repr=False)

    @property
    def n_links(self) -> int:
        return len(self.init_node)


def _fetch(network: str, filename: str) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    local = CACHE / filename
    if not local.exists():
        url = f"{RAW_BASE}/{network}/{filename}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            local.write_bytes(resp.read())
    return local.read_text()


def parse_net(text: str, name: str) -> TNTPNetwork:
    """Parse a ``*_net.tntp`` link table (metadata block + tilde-terminated rows)."""
    meta = dict(re.findall(r"<(\w+)>\s*([^\n<]+)", text))
    n_nodes = int(meta.get("NUMBER OF NODES".replace(" ", ""), 0) or 0)
    body = text.split("<END OF METADATA>")[-1]
    init, term, cap, fft, b_arr, power = [], [], [], [], [], []
    for line in body.splitlines():
        line = line.strip().rstrip(";").strip()
        if not line or line.startswith(("~", "<")):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        init.append(int(parts[0]))
        term.append(int(parts[1]))
        cap.append(float(parts[2]))
        fft.append(float(parts[4]))
        b_arr.append(float(parts[5]))
        power.append(float(parts[6]))
    if not n_nodes:
        n_nodes = max(max(init), max(term))
    return TNTPNetwork(
        name=name,
        n_nodes=n_nodes,
        init_node=tuple(init),
        term_node=tuple(term),
        capacity=tuple(cap),
        free_flow_time=tuple(fft),
        b=tuple(b_arr),
        power=tuple(power),
        demand={},
    )


def parse_trips(text: str) -> dict[tuple[int, int], float]:
    """Parse a ``*_trips.tntp`` OD matrix."""
    demand: dict[tuple[int, int], float] = {}
    origin = None
    for line in text.split("<END OF METADATA>")[-1].splitlines():
        line = line.strip()
        if not line or line.startswith("~"):
            continue
        if line.lower().startswith("origin"):
            origin = int(line.split()[1])
            continue
        if origin is None:
            continue
        for dest_str, value_str in re.findall(r"(\d+)\s*:\s*([\d.eE+-]+)\s*;", line):
            value = float(value_str)
            if value > 0 and int(dest_str) != origin:
                demand[(origin, int(dest_str))] = value
    return demand


def load_sioux_falls() -> TNTPNetwork:
    """Sioux Falls (24 zones, 76 links) — the debug-scale calibration network."""
    net = parse_net(_fetch("SiouxFalls", "SiouxFalls_net.tntp"), "SiouxFalls")
    demand = parse_trips(_fetch("SiouxFalls", "SiouxFalls_trips.tntp"))
    return TNTPNetwork(
        name=net.name,
        n_nodes=net.n_nodes,
        init_node=net.init_node,
        term_node=net.term_node,
        capacity=net.capacity,
        free_flow_time=net.free_flow_time,
        b=net.b,
        power=net.power,
        demand=demand,
    )


def load_best_known_flows() -> dict[tuple[int, int], float]:
    """Best-known UE link flows shipped with the repository (diagnostic anchor)."""
    text = _fetch("SiouxFalls", "SiouxFalls_flow.tntp")
    flows: dict[tuple[int, int], float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
            flows[(int(parts[0]), int(parts[1]))] = float(parts[2])
    return flows


def k_shortest_routes(
    net: TNTPNetwork, origin: int, destination: int, k: int
) -> list[tuple[int, ...]]:
    """k loop-less shortest paths by free-flow time (link-index sequences).

    Simple best-first enumeration (adequate at Sioux Falls scale); route-set
    restriction is a documented modelling choice, not hidden.
    """
    adjacency: dict[int, list[tuple[int, int]]] = {}
    for idx, (a, bnode) in enumerate(zip(net.init_node, net.term_node, strict=True)):
        adjacency.setdefault(a, []).append((bnode, idx))

    routes: list[tuple[int, ...]] = []
    heap: list[tuple[float, int, tuple[int, ...], frozenset[int]]] = [
        (0.0, origin, (), frozenset({origin}))
    ]
    counter = 0
    while heap and len(routes) < k and counter < 200_000:
        counter += 1
        cost, node, links, visited = heapq.heappop(heap)
        if node == destination:
            routes.append(links)
            continue
        for nxt, link_idx in adjacency.get(node, []):
            if nxt in visited:
                continue
            heapq.heappush(
                heap,
                (
                    cost + net.free_flow_time[link_idx],
                    nxt,
                    (*links, link_idx),
                    visited | {nxt},
                ),
            )
    return routes


def validate(net: TNTPNetwork) -> str:
    """Loud, specific validation report (Repository contract)."""
    problems = []
    if net.n_links == 0:
        problems.append("no links parsed")
    if any(c <= 0 for c in net.capacity):
        problems.append("non-positive capacities present")
    if any(t < 0 for t in net.free_flow_time):
        problems.append("negative free-flow times present")
    if not net.demand:
        problems.append("empty OD matrix — cannot assign any flow")
    total_demand = sum(net.demand.values())
    report = (
        f"{net.name}: {net.n_nodes} nodes, {net.n_links} links, "
        f"{len(net.demand)} OD pairs, total demand {total_demand:.0f}."
    )
    if problems:
        report += " REFUSING: " + "; ".join(problems)
    return report
