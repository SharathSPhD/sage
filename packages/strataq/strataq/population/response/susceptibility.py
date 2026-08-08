"""Population-engine response instruments: toll susceptibility and reciprocity.

The conjugate field here is the cleanest in the programme: **link tolls**
enter route costs exactly linearly (DOMAINS v1 §4.1). The SUE fixed point is
x = D·softmax(−θ(c(x) + Δτ)), so with S_pop = θ·blockdiag over OD of
D_od·C(σ_od) (the population choice covariance) and B_pop = DF(x) = −Dc,

    χ^pop = dx*/d(−Δτ) satisfies  χ = (I − S_pop B_pop … )⁻¹ S_pop Δ-layout

computed here directly on the route-flow tangent space (per-OD mean-zero).
For separable increasing link costs DF is symmetric ⟹ the toll response
matrix (routes × links, pulled back to routes × routes via Δ) is symmetric ⟹
reciprocity defect reads zero. The Hodge machinery does NOT transfer; DF
symmetry is the potentiality test (population/CLAUDE.md).

References
----------
Sandholm 2001/2010 (externality symmetry — cite, never claim); Fisk 1980;
Result 1's resolvent logic re-derived in the population state (tier: derived,
verified against finite differences).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from strataq.core.linalg import helmert_basis
from strataq.population.games.routing import (
    RoutingNetwork,
    payoff_field_jacobian,
    solve_sue,
)


def _od_blocks(net: RoutingNetwork) -> list[Array]:
    return [jnp.where(net.od_index == od)[0] for od in range(net.n_od)]


def population_choice_operator(net: RoutingNetwork, x: Array, theta: float) -> Array:
    """S_pop = θ · blockdiag_od( D_od C(σ_od) ) on route coordinates."""
    n = net.n_routes
    s_full = jnp.zeros((n, n))
    for od, idx in enumerate(_od_blocks(net)):
        share = x[idx] / net.demand[od]
        c_od = jnp.diag(share) - jnp.outer(share, share)
        s_full = s_full.at[jnp.ix_(idx, idx)].set(theta * net.demand[od] * c_od)
    return s_full


def population_tangent_basis(net: RoutingNetwork) -> Array:
    """Blockdiag Helmert basis of the per-OD mean-zero subspaces."""
    n = net.n_routes
    reduced = n - net.n_od
    q = jnp.zeros((n, reduced))
    col = 0
    for idx in _od_blocks(net):
        m = idx.shape[0]
        h = helmert_basis(int(m))
        q = q.at[jnp.ix_(idx, jnp.arange(col, col + m - 1))].set(h)
        col += m - 1
    return q


def toll_susceptibility(
    net: RoutingNetwork, theta: float, *, tolls: Array | None = None
) -> tuple[Array, Array]:
    """(χ_route, x*) — equilibrium response of route flows to route-cost shifts.

    χ is (n_routes, n_routes) on the per-OD tangent space lifted to full
    coordinates: dx*/dh with h a route-payoff (−cost) perturbation. Link-toll
    response is χ @ Δ (chain rule through the exactly-linear toll map).
    """
    x_star, _, _ = solve_sue(net, theta, tolls=tolls)
    s_pop = population_choice_operator(net, x_star, theta)
    b_pop = payoff_field_jacobian(net, x_star)
    q = population_tangent_basis(net)
    s_t = q.T @ s_pop @ q
    b_t = q.T @ b_pop @ q
    dim = s_t.shape[0]
    chi_t = jnp.linalg.solve(jnp.eye(dim) - s_t @ b_t, s_t)
    return q @ chi_t @ q.T, x_star


def population_reciprocity_defect(net: RoutingNetwork, theta: float) -> float:
    """ℛ of the toll-response matrix — must read 0: routing is exact potential."""
    chi, _ = toll_susceptibility(net, theta)
    asym = jnp.linalg.norm(chi - chi.T)
    sym = jnp.linalg.norm(chi + chi.T)
    return float(asym / sym)


def df_symmetry_defect(net: RoutingNetwork, x: Array) -> float:
    """The population potentiality test: ‖DF − DFᵀ‖/‖DF + DFᵀ‖ (0 = potential)."""
    df = payoff_field_jacobian(net, x)
    return float(jnp.linalg.norm(df - df.T) / jnp.linalg.norm(df + df.T))


def toll_susceptibility_fd(net: RoutingNetwork, theta: float, *, step: float = 1e-6) -> Array:
    """Finite-difference χ via route-cost perturbations (the oracle check)."""
    n = net.n_routes

    def flows_with_shift(route: int, sign: float) -> Array:
        shift = jnp.zeros(n).at[route].set(sign * step)
        x, _, _ = solve_sue(net, theta, route_cost_shift=shift, tol=1e-13)
        return x

    cols = []
    for route in range(n):
        # χ is dx/dh with h a payoff (= −cost) shift, so flip the sign.
        cols.append((flows_with_shift(route, -1.0) - flows_with_shift(route, +1.0)) / (2.0 * step))
    return jnp.stack(cols, axis=1)
