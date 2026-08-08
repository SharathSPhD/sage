"""Domain plugin 'blotto' — the α > 0 anchor of the calibration bracket.

Exactly the five contract objects on Engine 1; zero core changes. Payoffs are
known by construction; **budgets are the conjugate field** (experimenter-set,
exactly observable). Synthetic-only: round-level experimental data
availability is inconsistent (DOMAINS v1 §4.2), so ``loader`` is None until
data is actually in hand — no promised empirics.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array

from strataq.core.protocols import (
    ConjugateFieldSpec,
    DomainPlugin,
    LearnPageSpec,
)
from strataq.domains.blotto.oracle import BlottoOracle, allocations, blotto_game_tensors

ENGINE = "finite"


class BlottoGridBuilder:
    """ActionGridBuilder: enumerate integer allocations for each player's budget."""

    def __init__(self, budgets: Sequence[int], n_fields: int) -> None:
        self.budgets = tuple(int(b) for b in budgets)
        self.n_fields = int(n_fields)

    def build(self) -> tuple[Array, ...]:
        return tuple(
            jnp.asarray(allocations(b, self.n_fields), dtype=jnp.float64) for b in self.budgets
        )


FIELD = ConjugateFieldSpec(
    name="battlefield budget",
    observable=True,
    data_column=None,
    linearity="approximate",
    description=(
        "The experimenter-set per-player budget. A budget increment enlarges a "
        "player's allocation grid and shifts attainable payoffs — an exactly "
        "observable, exogenous perturbation (the cleanest field in the synthetic "
        "half of the programme). Linearity is 'approximate' because the payoff "
        "response to budget is a discrete envelope, not an additive term."
    ),
)

LEARN = LearnPageSpec(
    slug="blotto",
    title="Colonel Blotto — circulation you can see",
    controls=("lambda", "budget-a", "budget-b", "n-fields"),
)

PLUGIN = DomainPlugin(
    name="blotto",
    engine="finite",
    oracle_factory=BlottoOracle,
    grid_factory=BlottoGridBuilder,
    field_spec=FIELD,
    loader_factory=None,
    learn=LEARN,
)

__all__ = [
    "ENGINE",
    "FIELD",
    "LEARN",
    "PLUGIN",
    "BlottoGridBuilder",
    "BlottoOracle",
    "allocations",
    "blotto_game_tensors",
]
