"""Set a price against rivals who are also setting prices.

The problem is a discrete-price differentiated-products game: each firm picks a
level off a shared price grid, demand comes from the supplied model, and profit
is ``(price - cost) x quantity``. The equilibrium is the logit QRE of that game
at the given precision, so rivals are neither perfectly rational nor random —
which is what makes the rival price *distribution* an answer rather than a point
prediction.

``price`` is the profit-maximising level against that rival distribution, so with
a single firm the problem degenerates to the textbook monopoly price on the grid.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from strataq.core.solve.fixedpoint import logit_qre
from strataq.core.types import QREPoint
from strataq.finite.games.tensor import DenseTensorGame
from strataq.problems.base import (
    Diagnostics,
    GridSpec,
    Problem,
    Solution,
    Summary,
    as_float_vector,
    build_grid,
    check_convergence,
    check_profile_budget,
    finite_diagnostics,
    render,
)
from strataq.problems.demand import DemandModel

__all__ = ["PricingProblem", "PricingSolution"]

MAX_PROFILES = 250_000


@dataclass(frozen=True, repr=False, eq=False)
class PricingSolution(Solution):
    """The answer to a :class:`PricingProblem`."""

    price: float
    """The price to set for ``firm``."""
    profit: float
    """Expected profit at that price, against the equilibrium rival distribution."""
    price_grid: Array
    """The price levels considered, ``(n_levels,)``."""
    profit_curve: Array
    """Expected profit at every level of ``price_grid``, ``(n_levels,)``."""
    rival_prices: Array
    """Each rival's price distribution over ``price_grid``, ``(n_rivals, n_levels)``."""
    expected_rival_prices: Array
    """Each rival's mean price, ``(n_rivals,)``."""
    own_price_distribution: Array
    """This firm's own equilibrium mix, for reference, ``(n_levels,)``."""
    elasticities: Array
    """``E[i, j] = (dq_i/dp_j)(p_j/q_i)`` at the recommended price, ``(n_firms, n_firms)``."""
    costs: Array
    firm: int
    n_firms: int
    precision: float
    demand_model: str
    success: bool
    message: str
    game: DenseTensorGame = field(repr=False)
    point: QREPoint = field(repr=False)

    @cached_property
    def diagnostics(self) -> Diagnostics:
        """α, ℛ, ρ(SB), σ_EP — computed on first access, never on the solve path."""
        return finite_diagnostics(self.game, self.point)

    @property
    def margin(self) -> float:
        """Recommended price less this firm's marginal cost."""
        return self.price - float(self.costs[self.firm])

    def summary(self) -> Summary:
        own = float(self.elasticities[self.firm, self.firm])
        cross: float | None = None
        if self.n_firms > 1:
            off = [
                float(self.elasticities[self.firm, j])
                for j in range(self.n_firms)
                if j != self.firm
            ]
            cross = max(off)
        rival: float | None = None
        if self.expected_rival_prices.shape[0] > 0:
            rival = float(jnp.mean(self.expected_rival_prices))
        return render(
            "strataq PricingProblem",
            [
                ("price", self.price),
                ("profit", self.profit),
                ("margin", self.margin),
                ("own elasticity", own),
                ("cross elasticity", cross),
            ],
            [
                ("firms", self.n_firms),
                ("price levels", int(self.price_grid.shape[0])),
                ("precision", self.precision),
                ("demand", self.demand_model),
                ("mean rival price", rival),
            ],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "price": self.price,
            "profit": self.profit,
            "margin": self.margin,
            "price_grid": [float(p) for p in self.price_grid],
            "profit_curve": [float(p) for p in self.profit_curve],
            "rival_prices": [[float(p) for p in row] for row in self.rival_prices],
            "expected_rival_prices": [float(p) for p in self.expected_rival_prices],
            "own_price_distribution": [float(p) for p in self.own_price_distribution],
            "elasticities": [[float(e) for e in row] for row in self.elasticities],
            "costs": [float(c) for c in self.costs],
            "firm": self.firm,
            "n_firms": self.n_firms,
            "precision": self.precision,
            "demand_model": self.demand_model,
            "success": self.success,
            "message": self.message,
        }


class PricingProblem(Problem):
    """Price setting against rivals on a shared discrete grid.

    Parameters
    ----------
    costs
        Marginal cost per firm; a scalar with ``n_firms`` set means symmetric costs.
    grid
        ``(start, stop, step)`` or an explicit list of price levels.
    demand
        A :class:`~strataq.problems.demand.DemandModel`.
    precision
        Logit precision λ, in profit units. Larger means sharper best responses.
    firm
        Which firm the recommendation is for (default 0).
    """

    def __init__(
        self,
        *,
        costs: Sequence[float] | float,
        grid: GridSpec,
        demand: DemandModel,
        precision: float = 1.0,
        n_firms: int | None = None,
        firm: int = 0,
        tol: float | None = None,
        max_iter: int | None = None,
    ) -> None:
        raw = jnp.asarray(costs, dtype=jnp.float64).ravel()
        count = int(n_firms) if n_firms is not None else int(raw.shape[0])
        if count < 1:
            raise ValueError(f"n_firms must be >= 1, got {count}")
        self.costs = as_float_vector(raw, name="costs", length=count)
        self.price_grid = build_grid(grid, name="grid")
        if not isinstance(demand, DemandModel):
            raise ValueError(
                "demand must be a DemandModel (LogitDemand, LinearDemand or CustomDemand)"
            )
        demand.bind(count)
        if not float(precision) > 0:
            raise ValueError(f"precision must be > 0, got {precision}")
        if not 0 <= int(firm) < count:
            raise ValueError(f"firm must be in [0, {count}), got {firm}")
        check_profile_budget(int(self.price_grid.shape[0]), count, MAX_PROFILES, "PricingProblem")
        self.demand = demand
        self.n_firms = count
        self.precision = float(precision)
        self.firm = int(firm)
        self.tol = tol
        self.max_iter = max_iter

    def game(self) -> DenseTensorGame:
        """The profit tensors over the joint price grid — one per firm."""
        levels = self.price_grid
        n = self.n_firms
        mesh = jnp.stack(jnp.meshgrid(*([levels] * n), indexing="ij"), axis=-1)
        shape = mesh.shape[:-1]
        flat = mesh.reshape(-1, n)
        quantity = jax.vmap(self.demand.quantities)(flat)
        profit = (flat - self.costs[None, :]) * quantity
        return DenseTensorGame(tuple(profit[:, i].reshape(shape) for i in range(n)))

    def solve(self) -> PricingSolution:
        from strataq.core.defaults import base_config

        tol = base_config().tolerances.solve if self.tol is None else float(self.tol)
        game = self.game()
        point = logit_qre(game, self.precision, tol=self.tol, max_iter=self.max_iter)
        success, message = check_convergence(
            bool(point.converged), float(point.residual), tol, "PricingProblem.solve"
        )

        curve = point.expected_payoffs[self.firm]
        best = int(jnp.argmax(curve))
        price = float(self.price_grid[best])

        rivals = [i for i in range(self.n_firms) if i != self.firm]
        rival_mix = (
            jnp.stack([point.sigma[i] for i in rivals])
            if rivals
            else jnp.zeros((0, int(self.price_grid.shape[0])))
        )
        rival_mean = rival_mix @ self.price_grid

        evaluation = jnp.zeros((self.n_firms,), dtype=jnp.float64)
        evaluation = evaluation.at[self.firm].set(price)
        for slot, i in enumerate(rivals):
            evaluation = evaluation.at[i].set(rival_mean[slot])

        return PricingSolution(
            price=price,
            profit=float(curve[best]),
            price_grid=self.price_grid,
            profit_curve=curve,
            rival_prices=rival_mix,
            expected_rival_prices=rival_mean,
            own_price_distribution=point.sigma[self.firm],
            elasticities=self.demand.elasticities(evaluation),
            costs=self.costs,
            firm=self.firm,
            n_firms=self.n_firms,
            precision=self.precision,
            demand_model=self.demand.name,
            success=success,
            message=message,
            game=game,
            point=point,
        )
