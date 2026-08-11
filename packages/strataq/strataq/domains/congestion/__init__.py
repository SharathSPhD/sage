"""Domain plugin 'congestion' — the α = 0 anchor and the programme's calibration standard.

Engine 2 (population). BPR payoffs are given, the potential is the Beckmann
integral, logit route choice *is* Fisk's SUE — the only place where the
potential is known analytically AND real network data exists. **Link tolls are
the conjugate field**: they enter route costs exactly linearly, the cleanest
field anywhere in the programme (DOMAINS v1 §4.1).

Calibration vs empirics, stated once: computed SUE against the known potential
is calibration; empirical claims about actual driver dispersion would need
observed route shares, which TNTP does not provide.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array

from strataq.core.protocols import ConjugateFieldSpec, DomainPlugin, LearnPageSpec
from strataq.domains.congestion.tntp import (
    TNTPNetwork,
    k_shortest_routes,
    load_best_known_flows,
    load_sioux_falls,
    validate,
)
from strataq.population.games.routing import RoutingNetwork

ENGINE = "population"


def routing_network_from_tntp(
    net: TNTPNetwork,
    od_pairs: Sequence[tuple[int, int]],
    *,
    k_routes: int = 3,
) -> RoutingNetwork:
    """Build an explicit-route RoutingNetwork for chosen OD pairs.

    Route sets are the k shortest by free-flow time — a documented restriction
    (full route choice is exponential); instrument calibration (R = 0, DF
    symmetry, KKT) is exact *within* the route set regardless.
    """
    incidence_rows = []
    od_index = []
    demand = []
    for od_number, (origin, dest) in enumerate(od_pairs):
        routes = k_shortest_routes(net, origin, dest, k_routes)
        if not routes:
            raise ValueError(f"no route found for OD pair {(origin, dest)}")
        for links in routes:
            row = jnp.zeros(net.n_links)
            for link in links:
                row = row.at[link].set(1.0)
            incidence_rows.append(row)
            od_index.append(od_number)
        demand.append(net.demand[(origin, dest)])
    return RoutingNetwork(
        incidence=jnp.stack(incidence_rows),
        od_index=jnp.asarray(od_index, dtype=jnp.int32),
        demand=jnp.asarray(demand),
        free_flow=jnp.asarray(net.free_flow_time),
        b_coeff=jnp.asarray(net.b),
        capacity=jnp.asarray(net.capacity),
        power=jnp.asarray(net.power),
    )


class BPROracle:
    """PayoffOracle: payoff = −travel time on BPR links (population form)."""

    def __init__(self, network: RoutingNetwork) -> None:
        self.network = network
        self.n_players = 1  # a single population; OD pairs index sub-populations

    def profit(self, actions: Array, state: Array | None = None) -> Array:
        return -self.network.route_costs(actions)

    def quantity(self, actions: Array, state: Array | None = None) -> Array:
        return self.network.link_flows(actions)

    def response_matrix(self, actions: Array, state: Array | None = None) -> Array:
        from strataq.population.games.routing import payoff_field_jacobian

        return payoff_field_jacobian(self.network, actions)


class TNTPGridBuilder:
    """ActionGridBuilder: the per-OD route sets are the discrete action grid."""

    def __init__(self, net: TNTPNetwork, od_pairs: Sequence[tuple[int, int]], k_routes: int = 3):
        self.net = net
        self.od_pairs = tuple(od_pairs)
        self.k_routes = k_routes

    def build(self) -> tuple[Array, ...]:
        network = routing_network_from_tntp(self.net, self.od_pairs, k_routes=self.k_routes)
        return (network.incidence,)


class TNTPLoader:
    """DatasetLoader over the TransportationNetworks repository (cached fetch)."""

    def __init__(self, network: str = "SiouxFalls") -> None:
        if network != "SiouxFalls":
            raise ValueError(
                f"unsupported network {network!r}: SiouxFalls only for now "
                "(Anaheim, Chicago Sketch follow)"
            )
        self.network = network

    def load(self) -> TNTPNetwork:
        return load_sioux_falls()

    def validate(self) -> str:
        return validate(self.load())


FIELD = ConjugateFieldSpec(
    name="link tolls",
    observable=True,
    data_column="toll",
    linearity="exact",
    description=(
        "A toll on link l adds exactly tau_l to the cost of every route using l: "
        "an additive, exactly linear payoff perturbation — the cleanest conjugate "
        "field in the programme. Response instruments differentiate against it "
        "directly (population toll susceptibility)."
    ),
)

LEARN = LearnPageSpec(
    slug="congestion",
    title="Congestion — where the meters must read zero",
    controls=("theta", "toll-link", "toll-size", "od-pair"),
)

PLUGIN = DomainPlugin(
    name="congestion",
    engine="population",
    oracle_factory=BPROracle,
    grid_factory=TNTPGridBuilder,
    field_spec=FIELD,
    loader_factory=TNTPLoader,
    learn=LEARN,
)

__all__ = [
    "ENGINE",
    "FIELD",
    "LEARN",
    "PLUGIN",
    "BPROracle",
    "TNTPGridBuilder",
    "TNTPLoader",
    "load_best_known_flows",
    "load_sioux_falls",
    "routing_network_from_tntp",
]
