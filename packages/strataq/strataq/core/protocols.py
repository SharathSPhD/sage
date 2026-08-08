"""The plugin contract: the five objects every domain must ship, and nothing else.

A domain is a plugin iff it consists of exactly ``oracle``, ``grid``, ``field``,
``loader``, ``learn`` (plus an ``ENGINE`` tag) and requires zero changes to
``core/``, ``finite/`` or ``population/``. If a proposed domain needs new core
machinery, it is an engine, and engines require a recorded decision
(memory/decisions.md).

References
----------
THERMOQRE_DOMAINS_v1.md §5.2–§5.3 (the contract and the conjugate-field
requirement); THERMOQRE_PROGRAMME_v3.md §8.3 (PayoffOracle as the central
abstraction). Contract tier: engineering invariant, not a scientific claim.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol, runtime_checkable

from jax import Array

Engine = Literal["finite", "population", "bayesian"]
"""Which mathematical engine a domain runs on. ``bayesian`` is deferred."""


@runtime_checkable
class PayoffOracle(Protocol):
    """Maps a joint action profile (+ optional state) to per-player payoff.

    Wraps ANY payoff model — a demand system, BPR travel times, an ERCOT
    dispatch stack, a Blotto contest function. The library never assumes a
    functional form; it needs this map and, for gradient paths,
    JAX-differentiability.

    ``response_matrix`` is the (n_players, n_players) matrix of own- and
    cross-partial derivatives of payoff with respect to others' actions.
    Pricing plugins may expose ``elasticity`` as a domain-specific alias.
    """

    n_players: int

    def profit(self, actions: Array, state: Array | None = None) -> Array: ...

    def quantity(self, actions: Array, state: Array | None = None) -> Array: ...

    def response_matrix(self, actions: Array, state: Array | None = None) -> Array: ...


@runtime_checkable
class ActionGridBuilder(Protocol):
    """Builds a discrete per-player action grid from a continuous decision space.

    Builder pattern: configuration (bounds, resolution, empirical support)
    goes in at construction; ``build`` returns one grid per player.
    """

    def build(self) -> tuple[Array, ...]: ...


@dataclass(frozen=True)
class ConjugateFieldSpec:
    """What the observable payoff perturbation *h* is in a domain — or that there is none.

    The response instruments (``chi_equilibrium``, ``reciprocity_defect``) are
    derivatives with respect to an observable payoff perturbation. A domain
    that cannot name its field MUST declare ``ConjugateFieldSpec.NONE``; the
    library and API then refuse to compute those quantities for it rather than
    returning a meaningless number. This declaration is load-bearing: it is
    checked at registration, not documentation.

    Attributes
    ----------
    name:
        Human name of the field (e.g. "link tolls", "fuel cost shock",
        "battlefield budget", "wholesale cost shock").
    observable:
        False only on the ``NONE`` sentinel. Instruments key off this flag.
    data_column:
        Where to find the field in loaded panel data, if a loader exists.
    linearity:
        "exact" when h enters payoffs exactly linearly (tolls), "approximate"
        when linearisation is a modelling step (fuel shocks through dispatch).
    description:
        One-paragraph statement of how the perturbation enters payoffs.
    """

    name: str
    observable: bool = True
    data_column: str | None = None
    linearity: Literal["exact", "approximate"] = "exact"
    description: str = ""

    NONE: ClassVar[ConjugateFieldSpec]

    def __post_init__(self) -> None:
        if self.observable and not self.name:
            raise ValueError("An observable conjugate field must be named.")


# The sentinel: no observable payoff perturbation → response instruments unavailable.
ConjugateFieldSpec.NONE = ConjugateFieldSpec(
    name="",
    observable=False,
    linearity="exact",
    description="No observable conjugate field; response instruments are unavailable.",
)


@runtime_checkable
class DatasetLoader(Protocol):
    """Repository-pattern loader: a uniform interface over HF, TNTP, ERCOT, local files.

    ``load`` returns the domain's canonical panel (polars DataFrame in
    practice; typed as object here to keep core free of a hard polars
    dependency at the protocol level). ``validate`` returns a human-readable
    validation report and must refuse loudly and specifically when the data
    cannot identify what is being asked of it.
    """

    def load(self) -> object: ...

    def validate(self) -> str: ...


@dataclass(frozen=True)
class LearnPageSpec:
    """Pointer to a domain's Learn-mode explainer.

    Content lives once in ``docs/theory/`` and is rendered twice (docs site and
    app). The spec carries the slug and the interactive controls the page needs.
    """

    slug: str
    title: str
    controls: tuple[str, ...] = ()


@dataclass(frozen=True)
class DomainPlugin:
    """The assembled contract — what ``domains/<name>/__init__.py`` must export.

    Registration validates this object; a plugin with ``field=NONE`` registers
    successfully but with response instruments disabled.
    """

    name: str
    engine: Engine
    oracle_factory: Callable[..., PayoffOracle]
    grid_factory: Callable[..., ActionGridBuilder]
    field_spec: ConjugateFieldSpec
    loader_factory: Callable[..., DatasetLoader] | None
    learn: LearnPageSpec

    @property
    def response_instruments_available(self) -> bool:
        return self.field_spec.observable
