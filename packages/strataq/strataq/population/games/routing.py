"""Population routing games — Engine 2's concrete game object.

A continuum of travellers per OD pair distributes over routes; the state is a
route-flow vector x (not a profile of mixed strategies). Link costs are
separable and increasing (BPR); route cost = sum of its link costs. The payoff
field is F(x) = −c(x); its Jacobian DF = −Δᵀ diag(t'(v)) Δ is symmetric — the
population form of externality symmetry, so routing is an exact potential
population game with the Beckmann potential (K8).

References
----------
Beckmann–McGuire–Winsten 1956; Rosenthal 1973; Sandholm 2001/2010 (potential
population games); Fisk 1980 (SUE). Tier: exact (K8).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


class RoutingNetwork(eqx.Module):
    """Link-based network with explicit route sets (single or multi OD).

    ``incidence``: (n_routes, n_links) 0/1 matrix Δ (route uses link).
    ``od_index``: (n_routes,) int — which OD pair each route serves.
    ``demand``: (n_od,) total flow per OD pair.
    BPR link cost: t_l(v) = fft_l · (1 + b_l · (v / cap_l)^p_l).
    """

    incidence: Array
    od_index: Array
    demand: Array
    free_flow: Array  # (n_links,)
    b_coeff: Array  # (n_links,)
    capacity: Array  # (n_links,)
    power: Array  # (n_links,)

    @property
    def n_routes(self) -> int:
        return int(self.incidence.shape[0])

    @property
    def n_links(self) -> int:
        return int(self.incidence.shape[1])

    @property
    def n_od(self) -> int:
        return int(self.demand.shape[0])

    def link_flows(self, x: Array) -> Array:
        return self.incidence.T @ x

    def link_costs(self, v: Array) -> Array:
        return self.free_flow * (1.0 + self.b_coeff * (v / self.capacity) ** self.power)

    def route_costs(self, x: Array) -> Array:
        return self.incidence @ self.link_costs(self.link_flows(x))

    def beckmann(self, x: Array) -> Array:
        """The Beckmann integral Σ_l ∫₀^{v_l} t_l(u) du (closed form under BPR)."""
        v = self.link_flows(x)
        p = self.power
        integral = self.free_flow * (
            v + self.b_coeff * self.capacity / (p + 1.0) * (v / self.capacity) ** (p + 1.0)
        )
        return jnp.sum(integral)

    def fisk_objective(self, x: Array, theta: float) -> Array:
        """Beckmann + θ⁻¹ Σ x log x — the convex program whose optimum is SUE (K8)."""
        entropy_term = jnp.sum(jnp.where(x > 0, x * jnp.log(x), 0.0))
        return self.beckmann(x) + entropy_term / theta


def sue_logit_response(
    net: RoutingNetwork, x: Array, theta: float, tolls: Array | None = None
) -> Array:
    """One logit assignment sweep: per-OD softmax over generalised route costs."""
    costs = net.route_costs(x)
    if tolls is not None:
        costs = costs + net.incidence @ tolls
    out = jnp.zeros_like(x)
    for od in range(net.n_od):
        mask = net.od_index == od
        masked_cost = jnp.where(mask, -theta * costs, -jnp.inf)
        share = jnp.exp(jax.nn.log_softmax(masked_cost))
        out = out + net.demand[od] * jnp.where(mask, share, 0.0)
    return out


def solve_sue(
    net: RoutingNetwork,
    theta: float,
    *,
    tolls: Array | None = None,
    route_cost_shift: Array | None = None,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> tuple[Array, Array, Array]:
    """Fisk stochastic user equilibrium.

    Exploits convexity: damped Newton on the Fisk objective (Beckmann +
    θ⁻¹ entropy, plus exactly-linear toll/shift terms) in per-OD tangent
    coordinates. The Hessian Δᵀ t'(v) Δ + θ⁻¹ diag(1/x) is SPD on the tangent
    space, so Newton with backtracking converges globally and quadratically —
    the damped logit-assignment map is *not* a contraction on networks with
    steep BPR slopes (it two-cycles on the Braess diamond), which is why the
    solver goes through the convex program rather than the response map.

    Returns (route flows x*, fixed-point residual, Newton steps).
    """
    from strataq.core.linalg import helmert_basis

    generalised = jnp.zeros(net.n_routes)
    if tolls is not None:
        generalised = generalised + net.incidence @ tolls
    if route_cost_shift is not None:
        generalised = generalised + route_cost_shift

    # Per-OD tangent basis and uniform-share starting point.
    n = net.n_routes
    blocks = [jnp.where(net.od_index == od)[0] for od in range(net.n_od)]
    reduced = n - net.n_od
    q_basis = jnp.zeros((n, reduced))
    x0 = jnp.zeros(n)
    col = 0
    for od, idx in enumerate(blocks):
        m = int(idx.shape[0])
        q_basis = q_basis.at[jnp.ix_(idx, jnp.arange(col, col + m - 1))].set(helmert_basis(m))
        x0 = x0.at[idx].set(net.demand[od] / m)
        col += m - 1

    def objective(x: Array) -> Array:
        entropy_term = jnp.sum(jnp.where(x > 0, x * jnp.log(x), 0.0))
        return net.beckmann(x) + entropy_term / theta + x @ generalised

    def gradient(x: Array) -> Array:
        return net.route_costs(x) + generalised + (jnp.log(x) + 1.0) / theta

    def hessian(x: Array) -> Array:
        return -payoff_field_jacobian(net, x) + jnp.diag(1.0 / x) / theta

    x = x0
    steps = 0
    while steps < int(max_iter):
        steps += 1
        grad_t = q_basis.T @ gradient(x)
        if float(jnp.max(jnp.abs(grad_t))) < tol:
            break
        hess_t = q_basis.T @ hessian(x) @ q_basis
        direction = q_basis @ jnp.linalg.solve(hess_t, -grad_t)
        # Backtracking: stay strictly positive and decrease the objective.
        step = 1.0
        base = float(objective(x))
        for _ in range(60):
            candidate = x + step * direction
            if float(jnp.min(candidate)) > 0 and float(objective(candidate)) <= base:
                break
            step *= 0.5
        x = x + step * direction

    def respond(flows: Array) -> Array:
        costs = net.route_costs(flows) + generalised
        out = jnp.zeros_like(flows)
        for od in range(net.n_od):
            mask = net.od_index == od
            masked = jnp.where(mask, -theta * costs, -jnp.inf)
            share = jnp.exp(jax.nn.log_softmax(masked))
            out = out + net.demand[od] * jnp.where(mask, share, 0.0)
        return out

    residual = jnp.max(jnp.abs(respond(x) - x))
    return x, residual, jnp.asarray(steps)


def payoff_field_jacobian(net: RoutingNetwork, x: Array) -> Array:
    """DF(x) = −∂c/∂x = −Δᵀ diag(t'(v)) Δ … transposed layout (n_routes²).

    Symmetric for separable increasing link costs — the population
    externality-symmetry test (Sandholm; cite, never claim).
    """
    v = net.link_flows(x)
    t_prime = (
        net.free_flow
        * net.b_coeff
        * net.power
        / net.capacity
        * (v / net.capacity) ** (net.power - 1.0)
    )
    return -(net.incidence @ jnp.diag(t_prime) @ net.incidence.T)
