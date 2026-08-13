"""Choose a bid in a sealed-bid auction, or an offer in a procurement tender.

Bidders pick a level off a shared bid grid; the highest eligible bid wins the
sale (``values=``), or the lowest eligible offer wins the contract (``costs=``);
ties split. Payoff is ``value - bid`` to the winner of a sale and ``bid - cost``
to the winner of a tender, zero otherwise. The equilibrium is the logit QRE of
that finite game, which is what makes the rival-bid *distribution* — not a point
prediction — the object you actually plan against.

A reserve is a floor on an acceptable bid in a sale and a ceiling on an
acceptable offer in a tender; bids outside it never win.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal

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
    check_profile_budget,
    finite_diagnostics,
    render,
)

__all__ = ["AuctionProblem", "AuctionSolution"]

MAX_PROFILES = 250_000

Kind = Literal["sale", "procurement"]


@dataclass(frozen=True, repr=False, eq=False)
class AuctionSolution(Solution):
    """The answer to an :class:`AuctionProblem`."""

    bid: float
    """The bid (sale) or offer (tender) to submit."""
    surplus: float
    """Expected surplus at that bid, against the equilibrium rival distribution."""
    win_probability: float
    """Probability of winning at that bid."""
    bid_grid: Array
    surplus_curve: Array
    """Expected surplus at every level of ``bid_grid``."""
    win_curve: Array
    """Win probability at every level of ``bid_grid``."""
    rival_bids: Array
    """Each rival's bid distribution over ``bid_grid``, ``(n_rivals, n_levels)``."""
    own_bid_distribution: Array
    expected_clearing_bid: float
    """Expected winning bid across the whole equilibrium, when the lot is awarded."""
    valuation: float
    """This bidder's value (sale) or cost (tender)."""
    reserve: float | None
    kind: Kind
    bidder: int
    n_bidders: int
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
            "strataq AuctionProblem",
            [
                ("bid", self.bid),
                ("expected surplus", self.surplus),
                ("win probability", self.win_probability),
                ("margin if won", abs(self.valuation - self.bid)),
                ("clearing bid", self.expected_clearing_bid),
            ],
            [
                ("bidders", self.n_bidders),
                ("bid levels", int(self.bid_grid.shape[0])),
                ("reserve", self.reserve),
                ("format", self.kind),
                ("precision", self.precision),
            ],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "bid": self.bid,
            "surplus": self.surplus,
            "win_probability": self.win_probability,
            "bid_grid": [float(b) for b in self.bid_grid],
            "surplus_curve": [float(s) for s in self.surplus_curve],
            "win_curve": [float(w) for w in self.win_curve],
            "rival_bids": [[float(p) for p in row] for row in self.rival_bids],
            "own_bid_distribution": [float(p) for p in self.own_bid_distribution],
            "expected_clearing_bid": self.expected_clearing_bid,
            "valuation": self.valuation,
            "reserve": self.reserve,
            "kind": self.kind,
            "bidder": self.bidder,
            "n_bidders": self.n_bidders,
            "precision": self.precision,
            "success": self.success,
            "message": self.message,
        }


class AuctionProblem(Problem):
    """Sealed-bid sale (``values=``) or procurement tender (``costs=``).

    Parameters
    ----------
    values / costs
        Exactly one of the two. A scalar with ``n_bidders`` set means symmetric
        bidders; a sequence gives one entry per bidder.
    grid
        ``(start, stop, step)`` or an explicit list of bid levels.
    reserve
        Minimum acceptable bid (sale) or maximum acceptable offer (tender).
    precision
        Logit precision λ, in payoff units.
    """

    def __init__(
        self,
        *,
        grid: GridSpec,
        values: Sequence[float] | float | None = None,
        costs: Sequence[float] | float | None = None,
        n_bidders: int | None = None,
        reserve: float | None = None,
        precision: float = 1.0,
        bidder: int = 0,
        tol: float | None = None,
        max_iter: int | None = None,
    ) -> None:
        if (values is None) == (costs is None):
            raise ValueError("pass exactly one of values= (a sale) or costs= (a tender)")
        raw = jnp.asarray(values if values is not None else costs, dtype=jnp.float64).ravel()
        count = int(n_bidders) if n_bidders is not None else int(raw.shape[0])
        if count < 1:
            raise ValueError(f"n_bidders must be >= 1, got {count}")
        self.kind: Kind = "sale" if values is not None else "procurement"
        self.valuations = as_float_vector(
            raw, name="values" if values is not None else "costs", length=count
        )
        self.bid_grid = build_grid(grid, name="grid")
        if reserve is not None and not jnp.isfinite(jnp.asarray(float(reserve))):
            raise ValueError("reserve must be finite")
        if not float(precision) > 0:
            raise ValueError(f"precision must be > 0, got {precision}")
        if not 0 <= int(bidder) < count:
            raise ValueError(f"bidder must be in [0, {count}), got {bidder}")
        check_profile_budget(int(self.bid_grid.shape[0]), count, MAX_PROFILES, "AuctionProblem")
        self.n_bidders = count
        self.reserve = None if reserve is None else float(reserve)
        self.precision = float(precision)
        self.bidder = int(bidder)
        self.tol = tol
        self.max_iter = max_iter

    def _profiles(self) -> tuple[Array, tuple[int, ...]]:
        mesh = jnp.stack(jnp.meshgrid(*([self.bid_grid] * self.n_bidders), indexing="ij"), axis=-1)
        return mesh.reshape(-1, self.n_bidders), tuple(mesh.shape[:-1])

    def _win_probabilities(self, bids: Array) -> Array:
        """Per-profile probability that each bidder is awarded the lot (ties split)."""
        everyone = jnp.ones(bids.shape, dtype=bool)
        if self.kind == "sale":
            eligible = everyone if self.reserve is None else bids >= self.reserve
            masked = jnp.where(eligible, bids, -jnp.inf)
            extreme = jnp.max(masked, axis=1, keepdims=True)
        else:
            eligible = everyone if self.reserve is None else bids <= self.reserve
            masked = jnp.where(eligible, bids, jnp.inf)
            extreme = jnp.min(masked, axis=1, keepdims=True)
        winners = (eligible & (bids == extreme)).astype(jnp.float64)
        tied = jnp.sum(winners, axis=1, keepdims=True)
        return jnp.where(tied > 0.0, winners / jnp.maximum(tied, 1.0), 0.0)

    def games(self) -> tuple[DenseTensorGame, DenseTensorGame, Array]:
        """Payoff tensors, win-probability tensors, and the awarded-price tensor."""
        flat, shape = self._profiles()
        win = self._win_probabilities(flat)
        surplus = (
            (self.valuations[None, :] - flat) if self.kind == "sale" else (flat - self.valuations)
        )
        payoff = surplus * win
        price = jnp.sum(win * flat, axis=1)
        return (
            DenseTensorGame(tuple(payoff[:, i].reshape(shape) for i in range(self.n_bidders))),
            DenseTensorGame(tuple(win[:, i].reshape(shape) for i in range(self.n_bidders))),
            price.reshape(shape),
        )

    def solve(self) -> AuctionSolution:
        from strataq.core.defaults import base_config

        tol = base_config().tolerances.solve if self.tol is None else float(self.tol)
        game, win_game, price_tensor = self.games()
        point = logit_qre(game, self.precision, tol=self.tol, max_iter=self.max_iter)
        success, message = check_convergence(
            bool(point.converged), float(point.residual), tol, "AuctionProblem.solve"
        )

        curve = point.expected_payoffs[self.bidder]
        win_curve = expected_payoffs(win_game, point.sigma)[self.bidder]
        best = int(jnp.argmax(curve))

        joint = point.sigma[0]
        for i in range(1, self.n_bidders):
            joint = jnp.tensordot(joint, point.sigma[i], axes=0)
        clearing = float(jnp.sum(joint * price_tensor))

        rivals = [i for i in range(self.n_bidders) if i != self.bidder]
        rival_mix = (
            jnp.stack([point.sigma[i] for i in rivals])
            if rivals
            else jnp.zeros((0, int(self.bid_grid.shape[0])))
        )
        return AuctionSolution(
            bid=float(self.bid_grid[best]),
            surplus=float(curve[best]),
            win_probability=float(win_curve[best]),
            bid_grid=self.bid_grid,
            surplus_curve=curve,
            win_curve=win_curve,
            rival_bids=rival_mix,
            own_bid_distribution=point.sigma[self.bidder],
            expected_clearing_bid=clearing,
            valuation=float(self.valuations[self.bidder]),
            reserve=self.reserve,
            kind=self.kind,
            bidder=self.bidder,
            n_bidders=self.n_bidders,
            precision=self.precision,
            success=success,
            message=message,
            game=game,
            point=point,
        )
