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
from strataq.problems.evolutionary import EvolutionaryProblem, EvolutionarySolution
from strataq.problems.extensive import ExtensiveProblem, ExtensiveSolution
from strataq.problems.pricing import PricingProblem, PricingSolution
from strataq.problems.repeated import RepeatedProblem, RepeatedSolution
from strataq.problems.routing import RoutingProblem, RoutingSolution, TollEffect
from strataq.problems.situation import (
    Alternative,
    RivalView,
    Sensitivity,
    Situation,
    SituationSolution,
    solve_situation,
)

__all__ = [
    "AllocationProblem",
    "AllocationSolution",
    "Alternative",
    "AuctionProblem",
    "AuctionSolution",
    "ConvergenceWarning",
    "CustomDemand",
    "DemandModel",
    "Diagnostics",
    "ElectricityProblem",
    "ElectricitySolution",
    "EvolutionaryProblem",
    "EvolutionarySolution",
    "ExtensiveProblem",
    "ExtensiveSolution",
    "LinearDemand",
    "LogitDemand",
    "PricingProblem",
    "PricingSolution",
    "Problem",
    "RepeatedProblem",
    "RepeatedSolution",
    "RivalView",
    "RoutingProblem",
    "RoutingSolution",
    "Sensitivity",
    "Situation",
    "SituationSolution",
    "Solution",
    "Summary",
    "TollEffect",
    "solve_situation",
]
