"""Electricity domain: CAISO market data + a stylised bidding oracle.

Five-object contract (core/protocols.py): oracle (uniform-price auction),
grid (offer ladder), conjugate field (offer-price shift — enters payoffs
exactly linearly), loader (CAISO OASIS), learn spec. The irreversibility
readings on real data are findings F-0008/F-0009.
"""

from typing import Literal

from strataq.core.protocols import ConjugateFieldSpec, DomainPlugin, LearnPageSpec
from strataq.domains.electricity.caiso import (
    DEFAULT_NODE,
    discretize_quantile,
    fetch_dam_lmp,
    phase_embed,
)
from strataq.domains.electricity.oracle import (
    BiddingOracle,
    OfferGridBuilder,
    bidding_game,
    clearing_price_distribution,
)

ENGINE: Literal["finite", "population", "bayesian"] = "finite"

FIELD = ConjugateFieldSpec(
    name="offer-price shift",
    observable=True,
    data_column="lmp",
    linearity="exact",
    description=(
        "A shift h to one generator's offer at a given rung enters that "
        "rung's payoff exactly linearly (margin moves one-for-one when "
        "dispatched); the cross-response of the rival's mix to h is the "
        "domain's reciprocity measurement."
    ),
)

LEARN = LearnPageSpec(
    slug="10-the-same-machinery-everywhere",
    title="Electricity: the same machinery on a power market",
    controls=("lambda", "markup-ladder", "cost-spread"),
)

engine = ENGINE
oracle_factory = BiddingOracle
grid_factory = OfferGridBuilder
field_spec = FIELD
loader_factory = None  # panel loading is fetch_dam_lmp; DatasetLoader shim later
learn = LEARN

PLUGIN = DomainPlugin(
    name="electricity",
    engine=ENGINE,
    oracle_factory=BiddingOracle,
    grid_factory=OfferGridBuilder,
    field_spec=FIELD,
    loader_factory=None,
    learn=LEARN,
)

__all__ = [
    "DEFAULT_NODE",
    "ENGINE",
    "FIELD",
    "LEARN",
    "PLUGIN",
    "BiddingOracle",
    "OfferGridBuilder",
    "bidding_game",
    "clearing_price_distribution",
    "discretize_quantile",
    "fetch_dam_lmp",
    "phase_embed",
]
