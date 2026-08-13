"""Problem → ``.solve()`` → Solution: the task API over the strataq engine.

The conventions here are the Python scientific stack's, not this library's own
research vocabulary. A ``Problem`` holds a specification and validates it at
construction; ``.solve()`` returns an immutable ``Solution`` whose fields are
named for the domain (``price``, ``bid``, ``flows``) and never for the physics;
``.summary()`` renders a compact fixed-width table in the statsmodels idiom.

Response, decomposition and dissipation coordinates (α, ℛ, σ_EP) are reachable
through ``Solution.diagnostics`` and nowhere else, and they are computed lazily
so a solve costs only the solve.

Failure follows the same conventions: bad input raises ``ValueError`` with a
short actionable message; a solve that misses its tolerance sets
``success = False`` and emits a :class:`ConvergenceWarning`.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
from jax import Array

from strataq.core.types import QREPoint
from strataq.finite.games.tensor import DenseTensorGame

__all__ = [
    "ConvergenceWarning",
    "Diagnostics",
    "Problem",
    "Solution",
    "Summary",
]

_WIDTH = 62

Scalar = float | int | bool | str | None
Row = tuple[str, Scalar]
GridSpec = tuple[float, float, float] | Sequence[float] | Array


class ConvergenceWarning(UserWarning):
    """A solve returned without meeting its tolerance (scipy/statsmodels idiom)."""


class Summary(str):
    """A report that displays as itself in a REPL and prints with ``print``."""

    def __repr__(self) -> str:
        return str(self)


def _fmt(value: Scalar, digits: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}g}"


def render(title: str, left: Sequence[Row], right: Sequence[Row]) -> Summary:
    """Two-column fixed-width table: centred title, rule, rows, rule."""
    rule = "=" * _WIDTH
    lines = [title.center(_WIDTH).rstrip(), rule]
    for i in range(max(len(left), len(right))):
        lkey, lval = left[i] if i < len(left) else ("", None)
        rkey, rval = right[i] if i < len(right) else ("", None)
        left_cell = _fmt(lval) if lkey else ""
        right_cell = _fmt(rval) if rkey else ""
        lines.append(f"{lkey:<20}{left_cell:>13}   {rkey:<16}{right_cell:>10}".rstrip())
    lines.append(rule)
    return Summary("\n".join(lines))


class Diagnostics:
    """Engine read-outs, deliberately off the primary answer.

    ``residual`` and ``n_iter`` describe the solve. ``alpha``, ``reciprocity_defect``,
    ``entropy_production``, ``rho_sb``, ``distance_to_criticality`` and
    ``symmetry_defect`` are the library's response/decomposition/dissipation
    instruments; each is ``None`` where the problem class does not define it.
    """

    __slots__ = (
        "alpha",
        "distance_to_criticality",
        "entropy_production",
        "lambda_normalised",
        "n_iter",
        "reciprocity_defect",
        "residual",
        "rho_sb",
        "symmetry_defect",
    )

    def __init__(
        self,
        *,
        residual: float,
        n_iter: int,
        alpha: float | None = None,
        reciprocity_defect: float | None = None,
        entropy_production: float | None = None,
        rho_sb: float | None = None,
        distance_to_criticality: float | None = None,
        lambda_normalised: float | None = None,
        symmetry_defect: float | None = None,
    ) -> None:
        self.residual = residual
        self.n_iter = n_iter
        self.alpha = alpha
        self.reciprocity_defect = reciprocity_defect
        self.entropy_production = entropy_production
        self.rho_sb = rho_sb
        self.distance_to_criticality = distance_to_criticality
        self.lambda_normalised = lambda_normalised
        self.symmetry_defect = symmetry_defect

    def as_dict(self) -> dict[str, float | int | None]:
        return {name: getattr(self, name) for name in self.__slots__}

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in self.as_dict().items() if v is not None)
        return f"Diagnostics({body})"


class Solution:
    """Base class for every ``.solve()`` return value.

    Subclasses are frozen dataclasses with ``repr=False`` so that echoing a
    solution in a REPL prints its :meth:`summary` table.
    """

    success: bool
    message: str

    def summary(self) -> Summary:
        raise NotImplementedError

    def as_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self.summary())


class Problem:
    """Base class for every problem specification.

    Subclasses validate their inputs in ``__init__`` (``ValueError`` on bad input)
    and do all the work in :meth:`solve`.
    """

    def solve(self) -> Solution:
        raise NotImplementedError


def finite_diagnostics(
    game: DenseTensorGame, point: QREPoint, *, max_states: int = 400
) -> Diagnostics:
    """α, ℛ, ρ(SB) and σ_EP for a solved finite game — the optional read.

    ``σ_EP`` needs the dense Glauber generator over the joint profile space and is
    omitted (left ``None``) above ``max_states`` profiles rather than approximated.
    """
    alpha_value: float | None = None
    defect: float | None = None
    rho: float | None = None
    distance: float | None = None
    epr: float | None = None
    if game.n_players >= 2:
        from strataq.finite.decompose.hodge import alpha as harmonic_fraction
        from strataq.finite.response.reciprocity import reciprocity_defect
        from strataq.finite.response.susceptibility import chi_equilibrium

        response = chi_equilibrium(game, point)
        rho = float(response.rho_sb)
        distance = float(response.distance_to_criticality)
        alpha_value = float(harmonic_fraction(game))
        defect = float(reciprocity_defect(game, point, response=response))
        n_states = 1
        for m in game.num_actions:
            n_states *= m
        if n_states <= max_states:
            from strataq.thermo.exact import thermo_read

            epr = float(thermo_read(game, float(point.lam[0])).epr)
    return Diagnostics(
        residual=float(point.residual),
        n_iter=int(point.n_iter),
        alpha=alpha_value,
        reciprocity_defect=defect,
        entropy_production=epr,
        rho_sb=rho,
        distance_to_criticality=distance,
        lambda_normalised=float(point.lambda_normalised[0]),
    )


def check_convergence(converged: bool, residual: float, tol: float, what: str) -> tuple[bool, str]:
    """Ordinary-Python non-convergence: a flag, a message, a warning."""
    if converged:
        return True, "converged"
    message = (
        f"{what} did not converge: residual {residual:.3g} exceeds tolerance {tol:.3g}. "
        "Raise max_iter, lower precision, or coarsen the grid."
    )
    warnings.warn(message, ConvergenceWarning, stacklevel=3)
    return False, message


def build_grid(spec: GridSpec, *, name: str, min_points: int = 2) -> Array:
    """Action levels from ``(start, stop, step)`` or an explicit sequence.

    A 3-tuple is a range specification (inclusive of ``stop`` when it lands on the
    step); any other sequence is taken literally.
    """
    if isinstance(spec, tuple) and len(spec) == 3:
        start, stop, step = (float(v) for v in spec)
        if not all(math.isfinite(v) for v in (start, stop, step)):
            raise ValueError(f"{name} must be finite")
        if step <= 0:
            raise ValueError(f"{name} step must be > 0, got {step}")
        if stop < start:
            raise ValueError(f"{name} stop must be >= start, got ({start}, {stop})")
        count = math.floor((stop - start) / step + 1e-9) + 1
        values = jnp.asarray([start + i * step for i in range(count)], dtype=jnp.float64)
    else:
        values = jnp.asarray(spec, dtype=jnp.float64).ravel()
    if values.shape[0] < min_points:
        raise ValueError(f"{name} needs at least {min_points} levels, got {int(values.shape[0])}")
    if not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError(f"{name} must be finite")
    if not bool(jnp.all(jnp.diff(values) > 0)):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def check_profile_budget(n_actions: int, n_players: int, limit: int, what: str) -> None:
    """Guard the dense joint tensor before it is materialised."""
    size = n_actions**n_players
    if size > limit:
        raise ValueError(
            f"{what}: {n_players} players x {n_actions} levels = {size} joint profiles "
            f"exceeds the dense limit {limit}; coarsen the grid or reduce the players."
        )


def as_float_vector(values: Sequence[float] | float | Array, *, name: str, length: int) -> Array:
    """Broadcast a scalar or validate a sequence to ``length`` finite float64 entries."""
    array = jnp.asarray(values, dtype=jnp.float64).ravel()
    if array.shape[0] == 1 and length > 1:
        array = jnp.full((length,), array[0])
    if array.shape[0] != length:
        raise ValueError(f"{name} must have {length} entries, got {int(array.shape[0])}")
    if not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError(f"{name} must be finite")
    return array
