"""Colonel Blotto payoff oracle — payoffs known by construction.

Two players allocate integer budgets across k battlefields; a battlefield is
won by the larger allocation (ties split), and payoff is the total value of
battlefields won. The experimenter-set budgets are the domain's conjugate
field: shifting a player's budget is an observable payoff perturbation with
no estimation step in between — which is why Blotto anchors the α > 0 end of
the calibration bracket (DOMAINS v1 §4.2).

References
----------
Borel 1921; Roberson 2006 (equilibrium structure); DOMAINS v1 §4.2. Tier:
exact construction.
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp
from jax import Array


def allocations(budget: int, n_fields: int) -> tuple[tuple[int, ...], ...]:
    """All integer allocations of ``budget`` over ``n_fields`` (the action grid)."""
    if budget < 0 or n_fields < 1:
        raise ValueError("budget must be >= 0 and n_fields >= 1")
    out = []
    for cuts in itertools.combinations_with_replacement(range(budget + 1), n_fields - 1):
        parts = []
        prev = 0
        for c in (*cuts, budget):
            parts.append(c - prev)
            prev = c
        out.append(tuple(parts))
    return tuple(out)


class BlottoOracle:
    """PayoffOracle over joint allocation profiles (2-player, values per field)."""

    def __init__(self, field_values: Array) -> None:
        self.field_values = jnp.asarray(field_values, dtype=jnp.float64)
        self.n_players = 2

    def profit(self, actions: Array, state: Array | None = None) -> Array:
        """Payoff to each player for one joint profile.

        ``actions``: shape (2, k) integer allocations. Returns shape (2,).
        """
        a, b = actions[0], actions[1]
        win_a = jnp.where(a > b, 1.0, jnp.where(a == b, 0.5, 0.0))
        value_a = jnp.sum(win_a * self.field_values)
        total = jnp.sum(self.field_values)
        return jnp.stack([value_a, total - value_a])

    def quantity(self, actions: Array, state: Array | None = None) -> Array:
        """Battlefields won (ties count half), per player."""
        a, b = actions[0], actions[1]
        won_a = jnp.sum(jnp.where(a > b, 1.0, jnp.where(a == b, 0.5, 0.0)))
        return jnp.stack([won_a, self.field_values.shape[0] - won_a])

    def response_matrix(self, actions: Array, state: Array | None = None) -> Array:
        """Zero off a tie boundary: payoffs are locally constant in allocations.

        The meaningful response object in Blotto is the *budget* response
        (the conjugate field), not marginal allocation shifts; this returns
        the literal (2, 2) Jacobian of profit in total-allocation direction,
        which is zero a.e. — documented behaviour, not an omission.
        """
        return jnp.zeros((2, 2))


def blotto_game_tensors(
    oracle: BlottoOracle, budgets: tuple[int, int]
) -> tuple[Array, Array, tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Materialise the payoff tensors over the two players' allocation grids."""
    grid_a = allocations(budgets[0], int(oracle.field_values.shape[0]))
    grid_b = allocations(budgets[1], int(oracle.field_values.shape[0]))
    u_a = jnp.zeros((len(grid_a), len(grid_b)))
    u_b = jnp.zeros((len(grid_a), len(grid_b)))
    for i, a in enumerate(grid_a):
        for j, b in enumerate(grid_b):
            pay = oracle.profit(jnp.asarray([a, b], dtype=jnp.float64))
            u_a = u_a.at[i, j].set(pay[0])
            u_b = u_b.at[i, j].set(pay[1])
    return u_a, u_b, grid_a, grid_b
