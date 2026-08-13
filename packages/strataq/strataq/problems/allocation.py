"""Split a budget across contested fields — the Colonel Blotto problem.

Two sides allocate integer budgets over fields with known values; the larger
allocation takes a field (ties split it), and the payoff is the total value
captured. The equilibrium is the logit QRE over the full allocation grid, so the
answer is both a concrete allocation and the distribution the other side is
playing against you.

Uses the shipped :class:`~strataq.domains.blotto.oracle.BlottoOracle`, vectorised
over the joint allocation grid.
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
    Problem,
    Solution,
    Summary,
    check_convergence,
    finite_diagnostics,
    render,
)

__all__ = ["AllocationProblem", "AllocationSolution"]

MAX_PROFILES = 40_000


@dataclass(frozen=True, repr=False, eq=False)
class AllocationSolution(Solution):
    """The answer to an :class:`AllocationProblem`."""

    allocation: Array
    """The allocation to play, one integer per field, ``(n_fields,)``."""
    win_probability: float
    """Probability of capturing more than half the total field value (ties count half)."""
    expected_value: float
    """Expected value captured by that allocation against the rival's mix."""
    allocation_distribution: Array
    """Own equilibrium mix over ``allocations``, ``(n_own,)``."""
    rival_distribution: Array
    """Rival's equilibrium mix over ``rival_allocations``, ``(n_rival,)``."""
    allocations: Array
    """Own allocation grid, ``(n_own, n_fields)``."""
    rival_allocations: Array
    value_curve: Array
    """Expected value captured at every own allocation, ``(n_own,)``."""
    field_values: Array
    budget: int
    rival_budget: int
    n_fields: int
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
        spread = "/".join(str(int(a)) for a in self.allocation)
        return render(
            "strataq AllocationProblem",
            [
                ("allocation", spread),
                ("expected value", self.expected_value),
                ("win probability", self.win_probability),
                ("total at stake", float(jnp.sum(self.field_values))),
            ],
            [
                ("budget", self.budget),
                ("rival budget", self.rival_budget),
                ("fields", self.n_fields),
                ("precision", self.precision),
            ],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "allocation": [int(a) for a in self.allocation],
            "win_probability": self.win_probability,
            "expected_value": self.expected_value,
            "allocation_distribution": [float(p) for p in self.allocation_distribution],
            "rival_distribution": [float(p) for p in self.rival_distribution],
            "allocations": [[int(a) for a in row] for row in self.allocations],
            "rival_allocations": [[int(a) for a in row] for row in self.rival_allocations],
            "value_curve": [float(v) for v in self.value_curve],
            "field_values": [float(v) for v in self.field_values],
            "budget": self.budget,
            "rival_budget": self.rival_budget,
            "n_fields": self.n_fields,
            "precision": self.precision,
            "success": self.success,
            "message": self.message,
        }


class AllocationProblem(Problem):
    """Colonel-Blotto allocation of a budget across valued fields.

    Parameters
    ----------
    budget
        Your integer budget.
    field_values
        Value of each field; or give ``n_fields`` for equal unit values.
    rival_budget
        The other side's budget (defaults to yours).
    precision
        Logit precision λ, in value units.
    """

    def __init__(
        self,
        *,
        budget: int,
        field_values: Sequence[float] | Array | None = None,
        n_fields: int | None = None,
        rival_budget: int | None = None,
        precision: float = 1.0,
        tol: float | None = None,
        max_iter: int | None = None,
    ) -> None:
        if field_values is None:
            if n_fields is None:
                raise ValueError("pass field_values= or n_fields=")
            values = jnp.ones((int(n_fields),), dtype=jnp.float64)
        else:
            values = jnp.asarray(field_values, dtype=jnp.float64).ravel()
        if values.shape[0] < 2:
            raise ValueError(f"need at least 2 fields, got {int(values.shape[0])}")
        if n_fields is not None and int(n_fields) != int(values.shape[0]):
            raise ValueError(
                f"n_fields {n_fields} does not match field_values length {int(values.shape[0])}"
            )
        if not bool(jnp.all(values > 0)):
            raise ValueError("field_values must be > 0")
        if int(budget) < 1:
            raise ValueError(f"budget must be >= 1, got {budget}")
        other = int(budget) if rival_budget is None else int(rival_budget)
        if other < 1:
            raise ValueError(f"rival_budget must be >= 1, got {other}")
        if not float(precision) > 0:
            raise ValueError(f"precision must be > 0, got {precision}")
        self.field_values = values
        self.n_fields = int(values.shape[0])
        self.budget = int(budget)
        self.rival_budget = other
        self.precision = float(precision)
        self.tol = tol
        self.max_iter = max_iter

    def grids(self) -> tuple[Array, Array]:
        """The two sides' allocation grids, ``(n, n_fields)`` each."""
        from strataq.domains.blotto.oracle import allocations

        own = jnp.asarray(allocations(self.budget, self.n_fields), dtype=jnp.float64)
        rival = jnp.asarray(allocations(self.rival_budget, self.n_fields), dtype=jnp.float64)
        size = int(own.shape[0]) * int(rival.shape[0])
        if size > MAX_PROFILES:
            raise ValueError(
                f"AllocationProblem: {size} joint allocations exceeds the dense limit "
                f"{MAX_PROFILES}; lower the budgets or the number of fields."
            )
        return own, rival

    def game(self) -> tuple[DenseTensorGame, Array, Array]:
        """Value-captured tensors over the joint allocation grid, plus both grids."""
        from strataq.domains.blotto.oracle import BlottoOracle

        own, rival = self.grids()
        oracle = BlottoOracle(self.field_values)
        pairs = jnp.stack(jnp.broadcast_arrays(own[:, None, :], rival[None, :, :]), axis=2).reshape(
            -1, 2, self.n_fields
        )
        payoff = jax.vmap(oracle.profit)(pairs)
        shape = (int(own.shape[0]), int(rival.shape[0]))
        tensors = (payoff[:, 0].reshape(shape), payoff[:, 1].reshape(shape))
        return DenseTensorGame(tensors), own, rival

    def solve(self) -> AllocationSolution:
        from strataq.core.defaults import base_config

        tol = base_config().tolerances.solve if self.tol is None else float(self.tol)
        game, own, rival = self.game()
        point = logit_qre(game, self.precision, tol=self.tol, max_iter=self.max_iter)
        success, message = check_convergence(
            bool(point.converged), float(point.residual), tol, "AllocationProblem.solve"
        )

        curve = point.expected_payoffs[0]
        best = int(jnp.argmax(curve))
        half = float(jnp.sum(self.field_values)) / 2.0
        captured = game.payoffs[0]
        outcome = jnp.where(captured > half, 1.0, jnp.where(captured == half, 0.5, 0.0))
        win = float(outcome[best] @ point.sigma[1])

        return AllocationSolution(
            allocation=own[best].astype(jnp.int32),
            win_probability=win,
            expected_value=float(curve[best]),
            allocation_distribution=point.sigma[0],
            rival_distribution=point.sigma[1],
            allocations=own.astype(jnp.int32),
            rival_allocations=rival.astype(jnp.int32),
            value_curve=curve,
            field_values=self.field_values,
            budget=self.budget,
            rival_budget=self.rival_budget,
            n_fields=self.n_fields,
            precision=self.precision,
            success=success,
            message=message,
            game=game,
            point=point,
        )
