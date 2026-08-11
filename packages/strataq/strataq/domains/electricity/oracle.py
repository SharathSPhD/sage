"""Stylised uniform-price bidding game — the electricity domain's oracle.

Two symmetric generators, inelastic demand of one unit, discrete offer
prices: the cheaper offer dispatches and sets the clearing price (D=1 makes
pay-as-clear coincide with pay-as-bid at the margin); ties split the demand.
Undercutting steals dispatch, high offers harvest margin when the rival is
absent — a classic mixed potential/harmonic structure (α is measured, not
assumed).

Honesty tier: the ORACLE is exact for its own stylised rules; any λ estimated
by matching its QRE to real clearing-price dispersion is conditional on this
cost/demand model and is labelled as such wherever reported.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from strataq.finite.games.tensor import DenseTensorGame


def bidding_game(costs: tuple[float, float], offers: tuple[float, ...]) -> DenseTensorGame:
    """The 2-generator, D=1 uniform-price auction as a payoff tensor pair."""
    m = len(offers)
    u1 = np.zeros((m, m))
    u2 = np.zeros((m, m))
    for i, oi in enumerate(offers):
        for j, oj in enumerate(offers):
            if oi < oj:
                u1[i, j] = oi - costs[0]
            elif oj < oi:
                u2[i, j] = oj - costs[1]
            else:  # tie: split the unit at the common offer
                u1[i, j] = (oi - costs[0]) / 2.0
                u2[i, j] = (oj - costs[1]) / 2.0
    return DenseTensorGame((jnp.asarray(u1), jnp.asarray(u2)))


def clearing_price_distribution(
    sigma: tuple[Array, ...], offers: tuple[float, ...]
) -> tuple[list[float], list[float]]:
    """Distribution of the clearing price under a product mix (σ₁, σ₂).

    Clearing price = the dispatched (minimum) offer; ties clear at the tie.
    Returns (prices, probabilities) over the offer grid.
    """
    s1 = np.asarray(sigma[0])
    s2 = np.asarray(sigma[1])
    probs = np.zeros(len(offers))
    for i in range(len(offers)):
        for j in range(len(offers)):
            probs[min(i, j)] += s1[i] * s2[j]
    return list(offers), [float(p) for p in probs]


class BiddingOracle:
    """PayoffOracle over the stylised auction (protocol shim)."""

    def __init__(self, costs: tuple[float, float], offers: tuple[float, ...]) -> None:
        self.costs = costs
        self.offers = offers
        self.game = bidding_game(costs, offers)
        self.n_players = 2

    def profit(self, actions: Array, state: Array | None = None) -> Array:
        i, j = int(actions[0]), int(actions[1])
        return jnp.asarray([self.game.payoffs[0][i, j], self.game.payoffs[1][i, j]])

    def quantity(self, actions: Array, state: Array | None = None) -> Array:
        i, j = int(actions[0]), int(actions[1])
        if i == j:
            return jnp.asarray([0.5, 0.5])
        return jnp.asarray([1.0, 0.0]) if i < j else jnp.asarray([0.0, 1.0])

    def response_matrix(self, actions: Array, state: Array | None = None) -> Array:
        """(2, 2) Jacobian d profit_p / d offer_q on the discrete offer grid.

        Central differences in grid-index space (one rung of the markup
        ladder), clamped at the grid edges — the auction's payoff is a step
        function of the offer ORDER, so the grid step is the smallest
        meaningful perturbation. Added 2026-08-12: the protocol requires
        this method and the plugin shipped without it (caught by CI strict
        typing, not by any gate — see F-0018's lesson about blind spots).
        """
        idx = [int(actions[0]), int(actions[1])]
        n = len(self.offers)
        jac = jnp.zeros((2, 2))
        for q in range(2):  # perturb player q's offer index
            lo = list(idx)
            hi = list(idx)
            lo[q] = max(0, idx[q] - 1)
            hi[q] = min(n - 1, idx[q] + 1)
            span = hi[q] - lo[q]
            if span == 0:
                continue
            step = (self.offers[hi[q]] - self.offers[lo[q]]) / span
            if step == 0:
                continue
            d = (self.profit(jnp.asarray(hi)) - self.profit(jnp.asarray(lo))) / (span * step)
            jac = jac.at[:, q].set(d)
        return jac


class OfferGridBuilder:
    """ActionGridBuilder: offer grids as cost plus a markup ladder."""

    def __init__(self, costs: tuple[float, float], markups: tuple[float, ...]) -> None:
        self.costs = costs
        self.markups = markups

    def build(self) -> tuple[Array, ...]:
        return tuple(jnp.asarray([c + mk for mk in self.markups]) for c in self.costs)
