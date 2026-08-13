"""``strataq.repeated`` — infinitely repeated games with discounting.

Three things live here. :mod:`~strataq.repeated.automata` writes a repeated-game
strategy as a Moore machine, so grim trigger and tit-for-tat are data rather than
code paths. :mod:`~strataq.repeated.folk` answers the two questions the folk
theorem asks — is this profile sustainable at δ, and what is the critical δ —
both in closed form for grim trigger and generically by the one-shot-deviation
criterion, plus the logit analogue where sustainability is a probability.
:mod:`~strataq.repeated.cycles` runs alternating logit best response on a price
ladder and measures the Edgeworth cycle it produces.

References
----------
Friedman 1971; Abreu–Rubinstein 1988; Fudenberg–Maskin 1986; Maskin–Tirole 1988.
"""

from strataq.repeated.automata import Automaton, always, grim_trigger, tit_for_tat
from strataq.repeated.cycles import (
    PriceCycle,
    alternating_logit_path,
    bertrand_ladder,
    detect_cycle,
    edgeworth_cycle,
    linear_market_demand,
)
from strataq.repeated.folk import (
    MachineValues,
    RepeatedLogitPoint,
    SustainableSet,
    best_deviation,
    critical_discount,
    deviation_gains,
    grim_critical_discount,
    is_sustainable,
    logit_trigger_equilibrium,
    machine_values,
    minmax_payoffs,
    sustainable_payoff_set,
)

__all__ = [
    "Automaton",
    "MachineValues",
    "PriceCycle",
    "RepeatedLogitPoint",
    "SustainableSet",
    "alternating_logit_path",
    "always",
    "bertrand_ladder",
    "best_deviation",
    "critical_discount",
    "detect_cycle",
    "deviation_gains",
    "edgeworth_cycle",
    "grim_critical_discount",
    "grim_trigger",
    "is_sustainable",
    "linear_market_demand",
    "logit_trigger_equilibrium",
    "machine_values",
    "minmax_payoffs",
    "sustainable_payoff_set",
    "tit_for_tat",
]
