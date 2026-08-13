"""``strataq.problems`` — solve a real problem, get an answer with its numbers.

Every problem type follows the same three lines::

    prob = sq.PricingProblem(costs=..., grid=..., demand=..., precision=...)
    res = prob.solve()
    res.summary()

``Problem`` objects validate on construction and raise ``ValueError`` with a
short, actionable message. ``Solution`` objects are frozen, their fields are
named for the domain, and ``.summary()`` prints a compact table. Everything the
library knows about response, decomposition and dissipation sits on
``.diagnostics`` and is computed only if you ask for it.
"""

from strataq.problems.allocation import AllocationProblem, AllocationSolution
from strataq.problems.auction import AuctionProblem, AuctionSolution
from strataq.problems.base import (
    ConvergenceWarning,
    Diagnostics,
    Problem,
    Solution,
    Summary,
)
from strataq.problems.demand import CustomDemand, DemandModel, LinearDemand, LogitDemand
from strataq.problems.electricity import ElectricityProblem, ElectricitySolution
from strataq.problems.pricing import PricingProblem, PricingSolution
from strataq.problems.routing import RoutingProblem, RoutingSolution, TollEffect

__all__ = [
    "AllocationProblem",
    "AllocationSolution",
    "AuctionProblem",
    "AuctionSolution",
    "ConvergenceWarning",
    "CustomDemand",
    "DemandModel",
    "Diagnostics",
    "ElectricityProblem",
    "ElectricitySolution",
    "LinearDemand",
    "LogitDemand",
    "PricingProblem",
    "PricingSolution",
    "Problem",
    "RoutingProblem",
    "RoutingSolution",
    "Solution",
    "Summary",
    "TollEffect",
]
