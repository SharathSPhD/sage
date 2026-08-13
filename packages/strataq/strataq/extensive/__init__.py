"""``strataq.extensive`` — games as trees, not tables.

:mod:`~strataq.extensive.tree` is the representation: nodes, chance moves,
information sets and payoffs at leaves, stored as flat preorder arrays with a
nested-dict constructor that doubles as the JSON format.
:mod:`~strataq.extensive.behaviour` has the two tree passes everything is built
from, the reduced normal form, and Kuhn's theorem as a tested utility.
:mod:`~strataq.extensive.aqre` solves the agent QRE — logit at every information
set — on the same damped fixed point as the strategic-form solver.
:mod:`~strataq.extensive.backward` does backward induction on perfect-information
trees. :mod:`~strataq.extensive.catalogue` ships the classics.

References
----------
Kuhn 1953; Selten 1975; McKelvey–Palfrey, Experimental Economics 1998.
"""

from strataq.extensive.aqre import AQREPoint, agent_qre, agent_qre_branch
from strataq.extensive.backward import (
    BackwardInduction,
    backward_induction,
    subgame_perfect_behaviour,
)
from strataq.extensive.behaviour import (
    behaviour_to_mixed,
    expected_payoffs,
    mixed_to_behaviour,
    node_values,
    policy_from_behaviour,
    pure_strategies,
    reach_probabilities,
    realisation_gap,
    realisation_plan,
    reduced_normal_form,
    uniform_behaviour,
)
from strataq.extensive.catalogue import (
    CATALOGUE,
    bargaining,
    build,
    centipede,
    entry_deterrence,
    kuhn_poker,
    seltens_horse,
)
from strataq.extensive.tree import CHANCE, TERMINAL, ExtensiveGame, perfect_recall_violations

__all__ = [
    "CATALOGUE",
    "CHANCE",
    "TERMINAL",
    "AQREPoint",
    "BackwardInduction",
    "ExtensiveGame",
    "agent_qre",
    "agent_qre_branch",
    "backward_induction",
    "bargaining",
    "behaviour_to_mixed",
    "build",
    "centipede",
    "entry_deterrence",
    "expected_payoffs",
    "kuhn_poker",
    "mixed_to_behaviour",
    "node_values",
    "perfect_recall_violations",
    "policy_from_behaviour",
    "pure_strategies",
    "reach_probabilities",
    "realisation_gap",
    "realisation_plan",
    "reduced_normal_form",
    "seltens_horse",
    "subgame_perfect_behaviour",
    "uniform_behaviour",
]
