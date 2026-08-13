"""strataq.games — the calibration anchors, one import away.

A thin ergonomic namespace over :mod:`strataq.finite.games.library`. Everything here
returns a :class:`~strataq.finite.games.tensor.DenseTensorGame`, which every instrument
in the library (and :func:`strataq.diagnose`) accepts directly::

    >>> import strataq
    >>> rps = strataq.games.rock_paper_scissors()
    >>> rps.num_actions
    (3, 3)

Exact potential games (``congestion``, ``coordination``, ``common_interest``) are the
games the instruments must read *zero* on; the cyclic games (``rock_paper_scissors``,
``matching_pennies``) are the games they must read *positive* on. Nothing is
reimplemented here — this module only re-exports.

References
----------
Rosenthal 1973; Monderer–Shapley 1996; Candogan et al. 2011. Tier: exact.
"""

from __future__ import annotations

from strataq.finite.games.library import (
    common_interest,
    congestion,
    congestion_potential,
    coordination,
    matching_pennies,
    rock_paper_scissors,
)

__all__ = [
    "common_interest",
    "congestion",
    "congestion_potential",
    "coordination",
    "matching_pennies",
    "rock_paper_scissors",
]
