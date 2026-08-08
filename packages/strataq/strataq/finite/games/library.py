"""Verified game library: the calibration anchors.

Exact potential games (congestion, coordination, common-interest) — the
instruments must read zero on these. Pure harmonic games (RPS family,
matching pennies) — the instruments must read positive. Every constructor
documents its potential (or why none exists).

References
----------
Rosenthal 1973 (congestion potential); Monderer–Shapley 1996; Candogan et al.
2011 (harmonic examples). Tier: exact — these are textbook constructions.
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp
from jax import Array

from strataq.finite.games.tensor import DenseTensorGame


def common_interest(v: Array) -> DenseTensorGame:
    """All players receive v(a). Exact potential game with Φ = v."""
    v = jnp.asarray(v, dtype=jnp.float64)
    return DenseTensorGame(tuple(v for _ in range(v.ndim)))


def coordination(
    n_players: int, n_actions: int, bonus: float = 1.0, mismatch: float = 0.0
) -> DenseTensorGame:
    """Everyone-matches coordination: v(a) = bonus if all actions equal else mismatch.

    Common-interest, hence exact potential with Φ = v.
    """
    shape = (n_actions,) * n_players
    v = jnp.full(shape, float(mismatch))
    for a in range(n_actions):
        v = v.at[(a,) * n_players].set(float(bonus))
    return common_interest(v)


def congestion(n_players: int, resource_costs: Array) -> DenseTensorGame:
    """Rosenthal congestion game: each player picks one resource.

    ``resource_costs[r, k-1]`` is the cost of resource r when k players use it.
    Payoff to i is −cost of its chosen resource at its realised load. Exact
    potential Φ(a) = −Σ_r Σ_{k=1}^{load_r(a)} cost[r, k-1] (Rosenthal 1973).
    """
    costs = jnp.asarray(resource_costs, dtype=jnp.float64)
    n_resources = costs.shape[0]
    shape = (n_resources,) * n_players
    payoffs = [jnp.zeros(shape) for _ in range(n_players)]
    for profile in itertools.product(range(n_resources), repeat=n_players):
        loads = [profile.count(r) for r in range(n_resources)]
        for i in range(n_players):
            r = profile[i]
            payoffs[i] = payoffs[i].at[profile].set(-costs[r, loads[r] - 1])
    return DenseTensorGame(tuple(payoffs))


def congestion_potential(n_players: int, resource_costs: Array) -> Array:
    """The Rosenthal potential Φ(a) for :func:`congestion`, as a joint tensor."""
    costs = jnp.asarray(resource_costs, dtype=jnp.float64)
    n_resources = costs.shape[0]
    shape = (n_resources,) * n_players
    phi = jnp.zeros(shape)
    for profile in itertools.product(range(n_resources), repeat=n_players):
        loads = [profile.count(r) for r in range(n_resources)]
        val = -sum(float(jnp.sum(costs[r, : loads[r]])) for r in range(n_resources) if loads[r] > 0)
        phi = phi.at[profile].set(val)
    return phi


def rock_paper_scissors(n_actions: int = 3, win: float = 1.0) -> DenseTensorGame:
    """Two-player cyclic zero-sum game (odd n_actions ≥ 3): the pure harmonic anchor.

    A[a, b] = win if a beats b (cyclically), −win if b beats a, 0 on ties.
    Zero-sum with zero own-action means ⟹ zero potential component after
    normalisation: α = 1.
    """
    if n_actions < 3 or n_actions % 2 == 0:
        raise ValueError("cyclic RPS needs an odd number of actions >= 3")
    idx = jnp.arange(n_actions)
    diff = (idx[:, None] - idx[None, :]) % n_actions
    half = n_actions // 2
    a_matrix = jnp.where(
        (diff >= 1) & (diff <= half), float(win), jnp.where(diff == 0, 0.0, -float(win))
    )
    return DenseTensorGame((a_matrix, -a_matrix))


def matching_pennies(stake: float = 1.0) -> DenseTensorGame:
    """Two-player matching pennies: zero-sum, harmonic after normalisation."""
    a_matrix = jnp.array([[stake, -stake], [-stake, stake]], dtype=jnp.float64)
    return DenseTensorGame((a_matrix, -a_matrix))
