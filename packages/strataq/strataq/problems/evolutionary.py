"""Which behaviour spreads in a population, and does it stay?

The strategic-form question is "what would rational players do". The
evolutionary question is "what survives copying" — and the two have different
answers whenever a stable convention is not the efficient one. This problem
answers both at once: the rest points of the replicator with their stability, the
fixation probabilities of a mutant in a finite population, and — because the
selection intensity β is the logit precision λ — the strategic-form reading of
exactly the same game at exactly the same number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
from jax import Array

from strataq.evolutionary.moran import (
    IntensityComparison,
    compare_intensity,
    fixation_probability,
    moran_chain,
    pairwise_comparison_ratios,
)
from strataq.evolutionary.replicator import RestPoint, rest_points
from strataq.problems.base import (
    Problem,
    Solution,
    Summary,
    render,
)

__all__ = ["EvolutionaryProblem", "EvolutionarySolution"]

MAX_TYPES = 8
MAX_POPULATION = 5000


@dataclass(frozen=True, repr=False, eq=False)
class EvolutionarySolution(Solution):
    """The answer to an :class:`EvolutionaryProblem`."""

    rest_points: Array
    """Every replicator rest point, ``(k, n_types)``."""
    kinds: tuple[str, ...]
    """Stability classification of each rest point, in the same order."""
    max_real_parts: Array
    """Largest eigenvalue real part at each rest point, ``(k,)``."""
    stable: Array
    """The asymptotically stable rest points, ``(m, n_types)``."""
    intensity: float
    """Selection intensity β — the same number as the logit precision λ."""
    logit_rest_point: Array | None
    """``x = softmax(β A x)``, the logit-dynamic rest point, ``(n_types,)``."""
    qre_symmetric: Array | None
    """The symmetric logit QRE of the same game at λ = β, ``(n_types,)``."""
    qre_gap: float | None
    """Distance between the two — the evolutionary/finite reading agreement."""
    fermi_gap: float | None
    """Max gap between the Fermi imitation rule and the logit choice probability."""
    population: int | None
    fixation_a: float | None
    """Probability a single type-A mutant takes over a type-B population."""
    fixation_b: float | None
    moran_stationary: Array | None
    """Stationary distribution over the count of type A, ``(N+1,)``."""
    moran_share: float | None
    """``E[i/N]`` under that distribution."""
    monomorphic_weights: Array | None
    """Small-mutation weights on (all-B, all-A)."""
    success: bool
    message: str
    points: tuple[RestPoint, ...] = field(repr=False, default=())
    comparison: IntensityComparison | None = field(repr=False, default=None)

    def summary(self) -> Summary:
        return render(
            "strataq EvolutionaryProblem",
            [
                ("rest points", int(self.rest_points.shape[0])),
                ("stable", int(self.stable.shape[0])),
                ("intensity", self.intensity),
                ("qre gap", self.qre_gap),
                ("fermi gap", self.fermi_gap),
            ],
            [
                ("types", int(self.rest_points.shape[1]) if self.rest_points.size else 0),
                ("population", self.population),
                ("fixation A", self.fixation_a),
                ("fixation B", self.fixation_b),
                ("mean share", self.moran_share),
            ],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "rest_points": [[float(v) for v in row] for row in self.rest_points],
            "kinds": list(self.kinds),
            "max_real_parts": [float(v) for v in self.max_real_parts],
            "stable": [[float(v) for v in row] for row in self.stable],
            "intensity": self.intensity,
            "logit_rest_point": _optional_vector(self.logit_rest_point),
            "qre_symmetric": _optional_vector(self.qre_symmetric),
            "qre_gap": self.qre_gap,
            "fermi_gap": self.fermi_gap,
            "population": self.population,
            "fixation_a": self.fixation_a,
            "fixation_b": self.fixation_b,
            "moran_stationary": _optional_vector(self.moran_stationary),
            "moran_share": self.moran_share,
            "monomorphic_weights": _optional_vector(self.monomorphic_weights),
            "success": self.success,
            "message": self.message,
        }


def _optional_vector(values: Array | None) -> list[float] | None:
    return None if values is None else [float(v) for v in values]


class EvolutionaryProblem(Problem):
    """Replicator rest points, Moran fixation, and the β = λ comparison.

    Parameters
    ----------
    payoff
        A symmetric ``(n_types, n_types)`` payoff matrix: ``payoff[i, j]`` is what
        type ``i`` earns against type ``j``.
    intensity
        Selection intensity β for the finite population — and the logit precision
        λ for the strategic-form reading of the same game.
    population
        Population size ``N`` for the Moran chain. Required for the finite-
        population block, which two-type games get and larger games do not.
    mutation
        Mutation rate keeping the chain ergodic; defaults to ``config/base.yaml``.
    """

    def __init__(
        self,
        *,
        payoff: Sequence[Sequence[float]] | Array,
        intensity: float = 1.0,
        population: int | None = None,
        mutation: float | None = None,
        tol: float | None = None,
    ) -> None:
        matrix = jnp.asarray(payoff, dtype=jnp.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"payoff must be a square matrix, got {tuple(matrix.shape)}")
        if int(matrix.shape[0]) < 2:
            raise ValueError("need at least two types")
        if int(matrix.shape[0]) > MAX_TYPES:
            raise ValueError(
                f"rest-point enumeration is 2^n solves; {int(matrix.shape[0])} types exceeds "
                f"the limit {MAX_TYPES}"
            )
        if not bool(jnp.all(jnp.isfinite(matrix))):
            raise ValueError("payoff must be finite")
        if float(intensity) < 0:
            raise ValueError(f"intensity must be >= 0, got {intensity}")
        if population is not None:
            if int(population) < 2:
                raise ValueError(f"population must be >= 2, got {population}")
            if int(population) > MAX_POPULATION:
                raise ValueError(f"population exceeds the limit {MAX_POPULATION}")
            if int(matrix.shape[0]) != 2:
                raise ValueError(
                    "the Moran chain here is two-type; drop population= for a larger game"
                )
        self.payoff = matrix
        self.intensity = float(intensity)
        self.population = None if population is None else int(population)
        self.mutation = mutation
        self.tol = tol

    def solve(self) -> EvolutionarySolution:
        points = rest_points(self.payoff, tol=self.tol)
        states = (
            jnp.stack([p.x for p in points])
            if points
            else jnp.zeros((0, int(self.payoff.shape[0])))
        )
        kinds = tuple(p.kind for p in points)
        max_real = jnp.asarray([p.max_real_part for p in points], dtype=jnp.float64)
        stable = (
            jnp.stack([p.x for p in points if p.kind == "stable"])
            if any(p.kind == "stable" for p in points)
            else jnp.zeros((0, int(self.payoff.shape[0])))
        )

        comparison: IntensityComparison | None = None
        fixation_a: float | None = None
        fixation_b: float | None = None
        stationary: Array | None = None
        share: float | None = None
        weights: Array | None = None
        rest: Array | None = None
        qre: Array | None = None
        qre_gap: float | None = None
        fermi_gap: float | None = None

        if int(self.payoff.shape[0]) == 2 and self.population is not None:
            comparison = compare_intensity(
                self.payoff,
                self.intensity,
                self.population,
                mutation=self.mutation,
                tol=self.tol,
            )
            fixation_a = float(comparison.fixation_a)
            fixation_b = float(comparison.fixation_b)
            stationary = comparison.moran_stationary
            share = float(comparison.moran_share)
            weights = comparison.monomorphic_weights
            rest = comparison.logit_rest_point
            qre = comparison.qre_symmetric
            qre_gap = float(comparison.qre_gap)
            fermi_gap = float(comparison.fermi_gap)
        elif int(self.payoff.shape[0]) == 2:
            ratios = pairwise_comparison_ratios(self.payoff, 2, self.intensity)
            fixation_a = float(fixation_probability(ratios))

        return EvolutionarySolution(
            rest_points=states,
            kinds=kinds,
            max_real_parts=max_real,
            stable=stable,
            intensity=self.intensity,
            logit_rest_point=rest,
            qre_symmetric=qre,
            qre_gap=qre_gap,
            fermi_gap=fermi_gap,
            population=self.population,
            fixation_a=fixation_a,
            fixation_b=fixation_b,
            moran_stationary=stationary,
            moran_share=share,
            monomorphic_weights=weights,
            success=True,
            message=f"{len(points)} rest points, {int(stable.shape[0])} asymptotically stable",
            points=points,
            comparison=comparison,
        )


def moran_reading(
    payoff: Sequence[Sequence[float]] | Array,
    population: int,
    intensity: float,
    *,
    mutation: float | None = None,
) -> dict[str, Any]:
    """Fixation and stationarity for a two-type population, without the replicator."""
    chain = moran_chain(payoff, population, intensity, mutation=mutation)
    ratios = pairwise_comparison_ratios(payoff, population, intensity)
    rho_a = fixation_probability(ratios)
    return {
        "fixation_a": float(rho_a),
        "fixation_b": float(rho_a * jnp.exp(jnp.sum(jnp.log(ratios)))),
        "stationary": [float(p) for p in chain.stationary],
        "mean_share": float(chain.mean_share),
    }
