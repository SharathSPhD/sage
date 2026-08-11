"""Electricity domain: CAISO day-ahead market data through the instruments.

Loader-first slice (unit domains.electricity, plan-v2 R2): real DAM LMP
series → discretized price states → the trajectory irreversibility
estimators. Bidding-game oracle and the offer-price conjugate field follow
in the next slice.
"""

from strataq.domains.electricity.caiso import (
    DEFAULT_NODE,
    discretize_quantile,
    fetch_dam_lmp,
    phase_embed,
)

__all__ = ["DEFAULT_NODE", "discretize_quantile", "fetch_dam_lmp", "phase_embed"]
