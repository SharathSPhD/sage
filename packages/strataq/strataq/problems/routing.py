"""Assign traffic to a network and price a toll.

Travellers spread over routes according to logit route choice on BPR link costs;
the equilibrium is Fisk's stochastic user equilibrium, solved by the shipped
convex-program solver. Give it a network — the TNTP loader's Sioux Falls, a plain
edge list, or a :class:`~strataq.population.games.routing.RoutingNetwork` you
built yourself — plus OD demand, and it returns link flows, travel times, the
total cost of the assignment, and, when you toll links, exactly what the toll did.

Route sets are the ``k`` shortest paths by free-flow time per OD pair: a
documented restriction of the congestion plugin, not an approximation of it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import jax.numpy as jnp
from jax import Array

from strataq.population.games.routing import RoutingNetwork
from strataq.problems.base import (
    Diagnostics,
    Problem,
    Solution,
    Summary,
    check_convergence,
    render,
)

__all__ = ["RoutingProblem", "RoutingSolution", "TollEffect"]

SIOUX_FALLS = frozenset({"sioux_falls", "siouxfalls", "sioux falls"})

Edge = Mapping[str, float] | Sequence[float]
NetworkSpec = str | RoutingNetwork | Sequence[Edge]


@dataclass(frozen=True)
class TollEffect:
    """What the toll did, measured against the same network without it."""

    revenue: float
    delta_total_cost: float
    """Change in total travel time (negative means the toll reduced congestion)."""
    delta_flows: Array
    untolled_flows: Array
    tolled_links: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "revenue": self.revenue,
            "delta_total_cost": self.delta_total_cost,
            "delta_flows": [float(f) for f in self.delta_flows],
            "untolled_flows": [float(f) for f in self.untolled_flows],
            "tolled_links": list(self.tolled_links),
        }


@dataclass(frozen=True, repr=False, eq=False)
class RoutingSolution(Solution):
    """The answer to a :class:`RoutingProblem`."""

    flows: Array
    """Link flows, ``(n_links,)``."""
    travel_times: Array
    """Link travel times at those flows, ``(n_links,)``."""
    total_cost: float
    """Total travel time across the network (flow-weighted)."""
    route_flows: Array
    route_costs: Array
    toll_effect: TollEffect | None
    tolls: Array
    total_demand: float
    n_links: int
    n_routes: int
    n_od: int
    precision: float
    residual: float
    n_iter: int
    success: bool
    message: str
    network: RoutingNetwork = field(repr=False)

    @cached_property
    def diagnostics(self) -> Diagnostics:
        """Externality-symmetry defect and solver residuals — the population read."""
        from strataq.population.response.susceptibility import df_symmetry_defect

        return Diagnostics(
            residual=self.residual,
            n_iter=self.n_iter,
            symmetry_defect=float(df_symmetry_defect(self.network, self.route_flows)),
        )

    @property
    def mean_travel_time(self) -> float:
        """Total cost per traveller."""
        return self.total_cost / self.total_demand if self.total_demand > 0 else float("nan")

    def summary(self) -> Summary:
        saturation = float(jnp.max(self.flows / self.network.capacity))
        revenue = 0.0 if self.toll_effect is None else self.toll_effect.revenue
        change = None if self.toll_effect is None else self.toll_effect.delta_total_cost
        return render(
            "strataq RoutingProblem",
            [
                ("total cost", self.total_cost),
                ("mean travel time", self.mean_travel_time),
                ("max volume/capacity", saturation),
                ("toll revenue", revenue),
                ("cost change", change),
            ],
            [
                ("links", self.n_links),
                ("routes", self.n_routes),
                ("od pairs", self.n_od),
                ("precision", self.precision),
                ("converged", self.success),
            ],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "flows": [float(f) for f in self.flows],
            "travel_times": [float(t) for t in self.travel_times],
            "total_cost": self.total_cost,
            "mean_travel_time": self.mean_travel_time,
            "route_flows": [float(f) for f in self.route_flows],
            "route_costs": [float(c) for c in self.route_costs],
            "toll_effect": None if self.toll_effect is None else self.toll_effect.as_dict(),
            "tolls": [float(t) for t in self.tolls],
            "total_demand": self.total_demand,
            "n_links": self.n_links,
            "n_routes": self.n_routes,
            "n_od": self.n_od,
            "precision": self.precision,
            "residual": self.residual,
            "n_iter": self.n_iter,
            "success": self.success,
            "message": self.message,
        }


def _edge_fields(edge: Edge, index: int) -> tuple[int, int, float, float, float, float]:
    if isinstance(edge, Mapping):
        try:
            origin = int(edge["from"])
            destination = int(edge["to"])
            free_flow = float(edge["free_flow"])
            capacity = float(edge["capacity"])
        except KeyError as exc:
            raise ValueError(
                f"edge {index}: needs keys 'from', 'to', 'free_flow', 'capacity' "
                "(optional 'b', 'power')"
            ) from exc
        return (
            origin,
            destination,
            free_flow,
            capacity,
            float(edge.get("b", 0.15)),
            float(edge.get("power", 4.0)),
        )
    values = list(edge)
    if len(values) not in (4, 6):
        raise ValueError(
            f"edge {index}: expected (from, to, free_flow, capacity) or "
            f"(from, to, free_flow, capacity, b, power), got {len(values)} entries"
        )
    b, power = (0.15, 4.0) if len(values) == 4 else (float(values[4]), float(values[5]))
    return int(values[0]), int(values[1]), float(values[2]), float(values[3]), b, power


class RoutingProblem(Problem):
    """Traffic assignment with optional tolls.

    Parameters
    ----------
    network
        ``"sioux_falls"``, an edge list, or a ready ``RoutingNetwork``.
    demand
        ``{(origin, destination): trips}``. Required for an edge list; for Sioux
        Falls it defaults to the ``max_od`` busiest OD pairs of the TNTP matrix.
    tolls
        ``{link_index: toll}`` or a full vector of length ``n_links``.
    precision
        Logit route-choice precision θ. Larger means travellers spread less.
    """

    def __init__(
        self,
        *,
        network: NetworkSpec,
        demand: Mapping[tuple[int, int], float] | None = None,
        tolls: Mapping[int, float] | Sequence[float] | Array | None = None,
        precision: float = 0.5,
        k_routes: int = 3,
        max_od: int = 12,
        tol: float = 1e-12,
        max_iter: int = 200,
        residual_tol: float = 1e-8,
    ) -> None:
        if not float(precision) > 0:
            raise ValueError(f"precision must be > 0, got {precision}")
        if int(k_routes) < 1:
            raise ValueError(f"k_routes must be >= 1, got {k_routes}")
        if int(max_iter) < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        self.precision = float(precision)
        self.k_routes = int(k_routes)
        self.max_od = int(max_od)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.residual_tol = float(residual_tol)
        self.network = self._build_network(network, demand)
        self.tolls = self._build_tolls(tolls)

    def _build_network(
        self, network: NetworkSpec, demand: Mapping[tuple[int, int], float] | None
    ) -> RoutingNetwork:
        if isinstance(network, RoutingNetwork):
            if demand is not None:
                raise ValueError("a RoutingNetwork already carries its OD demand; drop demand=")
            return network
        from strataq.domains.congestion import routing_network_from_tntp
        from strataq.domains.congestion.tntp import TNTPNetwork, load_sioux_falls

        if isinstance(network, str):
            if network.strip().lower() not in SIOUX_FALLS:
                raise ValueError(
                    f"unknown network {network!r}: pass 'sioux_falls', an edge list, "
                    "or a RoutingNetwork"
                )
            tntp = load_sioux_falls()
            pairs = (
                list(demand)
                if demand is not None
                else sorted(tntp.demand, key=lambda od: tntp.demand[od], reverse=True)[
                    : self.max_od
                ]
            )
            if demand is not None:
                tntp = TNTPNetwork(
                    name=tntp.name,
                    n_nodes=tntp.n_nodes,
                    init_node=tntp.init_node,
                    term_node=tntp.term_node,
                    capacity=tntp.capacity,
                    free_flow_time=tntp.free_flow_time,
                    b=tntp.b,
                    power=tntp.power,
                    demand={od: float(q) for od, q in demand.items()},
                )
            return routing_network_from_tntp(tntp, pairs, k_routes=self.k_routes)

        edges = list(network)
        if not edges:
            raise ValueError("network edge list is empty")
        if not demand:
            raise ValueError("an edge-list network needs demand={(origin, dest): trips}")
        parsed = [_edge_fields(edge, i) for i, edge in enumerate(edges)]
        for index, (_, _, free_flow, capacity, _, _) in enumerate(parsed):
            if capacity <= 0:
                raise ValueError(f"edge {index}: capacity must be > 0, got {capacity}")
            if free_flow < 0:
                raise ValueError(f"edge {index}: free_flow must be >= 0, got {free_flow}")
        nodes = {node for origin, destination, *_ in parsed for node in (origin, destination)}
        tntp = TNTPNetwork(
            name="edge-list",
            n_nodes=max(nodes),
            init_node=tuple(e[0] for e in parsed),
            term_node=tuple(e[1] for e in parsed),
            capacity=tuple(e[3] for e in parsed),
            free_flow_time=tuple(e[2] for e in parsed),
            b=tuple(e[4] for e in parsed),
            power=tuple(e[5] for e in parsed),
            demand={od: float(q) for od, q in demand.items()},
        )
        for (origin, destination), trips in tntp.demand.items():
            if origin not in nodes or destination not in nodes:
                raise ValueError(f"OD pair {(origin, destination)} uses a node not in the edges")
            if trips <= 0:
                raise ValueError(f"OD pair {(origin, destination)} needs trips > 0")
        return routing_network_from_tntp(tntp, list(tntp.demand), k_routes=self.k_routes)

    def _build_tolls(
        self, tolls: Mapping[int, float] | Sequence[float] | Array | None
    ) -> Array | None:
        if tolls is None:
            return None
        n_links = self.network.n_links
        if isinstance(tolls, Mapping):
            vector = jnp.zeros((n_links,), dtype=jnp.float64)
            for link, value in tolls.items():
                if not 0 <= int(link) < n_links:
                    raise ValueError(f"toll link {link} outside [0, {n_links})")
                vector = vector.at[int(link)].set(float(value))
            return vector
        vector = jnp.asarray(tolls, dtype=jnp.float64).ravel()
        if vector.shape[0] != n_links:
            raise ValueError(
                f"tolls vector has {int(vector.shape[0])} entries, network has {n_links} links"
            )
        return vector

    def solve(self) -> RoutingSolution:
        from strataq.population.games.routing import solve_sue

        net = self.network
        flows, residual, steps = solve_sue(
            net, self.precision, tolls=self.tolls, tol=self.tol, max_iter=self.max_iter
        )
        success, message = check_convergence(
            float(residual) < self.residual_tol,
            float(residual),
            self.residual_tol,
            "RoutingProblem.solve",
        )
        link_flows = net.link_flows(flows)
        link_costs = net.link_costs(link_flows)
        total_cost = float(link_flows @ link_costs)

        effect: TollEffect | None = None
        if self.tolls is not None:
            base_flows, _, _ = solve_sue(net, self.precision, tol=self.tol, max_iter=self.max_iter)
            base_links = net.link_flows(base_flows)
            base_cost = float(base_links @ net.link_costs(base_links))
            effect = TollEffect(
                revenue=float(link_flows @ self.tolls),
                delta_total_cost=total_cost - base_cost,
                delta_flows=link_flows - base_links,
                untolled_flows=base_links,
                tolled_links=tuple(int(i) for i in jnp.nonzero(self.tolls)[0]),
            )

        return RoutingSolution(
            flows=link_flows,
            travel_times=link_costs,
            total_cost=total_cost,
            route_flows=flows,
            route_costs=net.route_costs(flows),
            toll_effect=effect,
            tolls=jnp.zeros((net.n_links,)) if self.tolls is None else self.tolls,
            total_demand=float(jnp.sum(net.demand)),
            n_links=net.n_links,
            n_routes=net.n_routes,
            n_od=net.n_od,
            precision=self.precision,
            residual=float(residual),
            n_iter=int(steps),
            success=success,
            message=message,
            network=net,
        )
