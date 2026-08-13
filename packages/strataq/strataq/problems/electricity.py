"""Offer a generator into a uniform-price market.

Two generators offer a block into an inelastic demand; the cheaper offer
dispatches and sets the clearing price, ties split the block. That is the shipped
:mod:`strataq.domains.electricity` oracle, and the equilibrium over the offer
ladder is its logit QRE — so the answer is an offer *curve* (a probability over
offer prices) plus the clearing-price distribution it implies, which is what a
trader actually needs to see.

The oracle dispatches a single marginal block, so ``demand`` must be within the
smaller generator's capacity; profits scale linearly with the block size.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import jax.numpy as jnp
from jax import Array

from strataq.core.solve.fixedpoint import logit_qre
from strataq.core.types import QREPoint
from strataq.finite.games.tensor import DenseTensorGame, expected_payoffs
from strataq.problems.base import (
    Diagnostics,
    GridSpec,
    Problem,
    Solution,
    Summary,
    as_float_vector,
    build_grid,
    check_convergence,
    finite_diagnostics,
    render,
)

__all__ = ["ElectricityProblem", "ElectricitySolution"]

MAX_OFFERS = 200


@dataclass(frozen=True, repr=False, eq=False)
class ElectricitySolution(Solution):
    """The answer to an :class:`ElectricityProblem`."""

    offer: float
    """The single best offer price, if you have to name one."""
    offer_curve: Array
    """The offer ladder to submit: ``(n_offers, 2)`` of ``[price, probability]``."""
    clearing_price: float
    """Expected clearing price of the market."""
    clearing_price_distribution: Array
    """``(n_offers, 2)`` of ``[price, probability]`` for the market clearing price."""
    revenue: float
    """Expected revenue for this generator (price x dispatched energy)."""
    profit: float
    """Expected profit for this generator at ``offer``."""
    dispatch_probability: float
    """Probability this generator is dispatched at ``offer``."""
    profit_curve: Array
    offers: Array
    costs: Array
    capacities: Array
    demand: float
    generator: int
    precision: float
    success: bool
    message: str
    game: DenseTensorGame = field(repr=False)
    point: QREPoint = field(repr=False)

    @cached_property
    def diagnostics(self) -> Diagnostics:
        """α, ℛ, ρ(SB), σ_EP — computed on first access, never on the solve path."""
        return finite_diagnostics(self.game, self.point)

    def summary(self) -> Summary:
        return render(
            "strataq ElectricityProblem",
            [
                ("offer", self.offer),
                ("clearing price", self.clearing_price),
                ("revenue", self.revenue),
                ("profit", self.profit),
                ("dispatch prob", self.dispatch_probability),
            ],
            [
                ("generators", int(self.costs.shape[0])),
                ("offer levels", int(self.offers.shape[0])),
                ("demand", self.demand),
                ("marginal cost", float(self.costs[self.generator])),
                ("precision", self.precision),
            ],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "offer": self.offer,
            "offer_curve": [[float(p), float(q)] for p, q in self.offer_curve],
            "clearing_price": self.clearing_price,
            "clearing_price_distribution": [
                [float(p), float(q)] for p, q in self.clearing_price_distribution
            ],
            "revenue": self.revenue,
            "profit": self.profit,
            "dispatch_probability": self.dispatch_probability,
            "profit_curve": [float(p) for p in self.profit_curve],
            "offers": [float(o) for o in self.offers],
            "costs": [float(c) for c in self.costs],
            "capacities": [float(c) for c in self.capacities],
            "demand": self.demand,
            "generator": self.generator,
            "precision": self.precision,
            "success": self.success,
            "message": self.message,
        }


class ElectricityProblem(Problem):
    """Offer strategy in a two-generator uniform-price market.

    Parameters
    ----------
    costs
        Marginal cost of each of the two generators.
    offers
        ``(start, stop, step)`` or an explicit list of offer prices.
    capacities
        Each generator's capacity; both must cover ``demand``.
    demand
        Inelastic demand for the interval (the block that clears).
    precision
        Logit precision λ, in profit units.
    """

    def __init__(
        self,
        *,
        costs: Sequence[float],
        offers: GridSpec,
        capacities: Sequence[float] | float | None = None,
        demand: float = 1.0,
        precision: float = 1.0,
        generator: int = 0,
        tol: float | None = None,
        max_iter: int | None = None,
    ) -> None:
        marginal = jnp.asarray(costs, dtype=jnp.float64).ravel()
        if marginal.shape[0] != 2:
            raise ValueError(
                f"the shipped electricity oracle models two generators; costs has "
                f"{int(marginal.shape[0])} entries"
            )
        self.costs = as_float_vector(marginal, name="costs", length=2)
        self.offers = build_grid(offers, name="offers")
        if int(self.offers.shape[0]) > MAX_OFFERS:
            raise ValueError(f"offers has {int(self.offers.shape[0])} levels; cap is {MAX_OFFERS}")
        if not float(demand) > 0:
            raise ValueError(f"demand must be > 0, got {demand}")
        caps = (
            jnp.full((2,), float(demand))
            if capacities is None
            else as_float_vector(capacities, name="capacities", length=2)
        )
        if float(jnp.min(caps)) < float(demand):
            raise ValueError(
                f"demand {demand} exceeds the smaller capacity {float(jnp.min(caps))}; the "
                "shipped oracle dispatches one marginal block, so both units must cover it."
            )
        if not float(precision) > 0:
            raise ValueError(f"precision must be > 0, got {precision}")
        if generator not in (0, 1):
            raise ValueError(f"generator must be 0 or 1, got {generator}")
        self.capacities = caps
        self.demand = float(demand)
        self.precision = float(precision)
        self.generator = int(generator)
        self.tol = tol
        self.max_iter = max_iter

    def game(self) -> DenseTensorGame:
        """The bidding game from the electricity oracle, scaled by the block size."""
        from strataq.domains.electricity.oracle import bidding_game

        base = bidding_game(
            (float(self.costs[0]), float(self.costs[1])),
            tuple(float(o) for o in self.offers),
        )
        return DenseTensorGame(tuple(u * self.demand for u in base.payoffs))

    def solve(self) -> ElectricitySolution:
        from strataq.core.defaults import base_config
        from strataq.domains.electricity.oracle import clearing_price_distribution

        tol = base_config().tolerances.solve if self.tol is None else float(self.tol)
        game = self.game()
        point = logit_qre(game, self.precision, tol=self.tol, max_iter=self.max_iter)
        success, message = check_convergence(
            bool(point.converged), float(point.residual), tol, "ElectricityProblem.solve"
        )

        curve = point.expected_payoffs[self.generator]
        best = int(jnp.argmax(curve))
        prices, probabilities = clearing_price_distribution(
            point.sigma, tuple(float(o) for o in self.offers)
        )
        price_array = jnp.asarray(prices, dtype=jnp.float64)
        probability_array = jnp.asarray(probabilities, dtype=jnp.float64)
        clearing = float(price_array @ probability_array)

        dispatch_tensors = self._dispatch_tensors()
        dispatch = expected_payoffs(dispatch_tensors, point.sigma)[self.generator]
        revenue_tensors = self._revenue_tensors()
        revenue_curve = expected_payoffs(revenue_tensors, point.sigma)[self.generator]

        return ElectricitySolution(
            offer=float(self.offers[best]),
            offer_curve=jnp.stack([self.offers, point.sigma[self.generator]], axis=1),
            clearing_price=clearing,
            clearing_price_distribution=jnp.stack([price_array, probability_array], axis=1),
            revenue=float(revenue_curve[best]),
            profit=float(curve[best]),
            dispatch_probability=float(dispatch[best]),
            profit_curve=curve,
            offers=self.offers,
            costs=self.costs,
            capacities=self.capacities,
            demand=self.demand,
            generator=self.generator,
            precision=self.precision,
            success=success,
            message=message,
            game=game,
            point=point,
        )

    def _dispatch_share(self) -> Array:
        """``(m, m)`` share of the block each generator serves, generator 0 first."""
        own = self.offers[:, None]
        other = self.offers[None, :]
        share_0 = jnp.where(own < other, 1.0, jnp.where(own == other, 0.5, 0.0))
        return jnp.stack([share_0, 1.0 - share_0])

    def _dispatch_tensors(self) -> DenseTensorGame:
        share = self._dispatch_share()
        return DenseTensorGame((share[0], share[1]))

    def _revenue_tensors(self) -> DenseTensorGame:
        share = self._dispatch_share()
        cleared = jnp.minimum(self.offers[:, None], self.offers[None, :])
        energy = self.demand * cleared
        return DenseTensorGame((share[0] * energy, share[1] * energy))
