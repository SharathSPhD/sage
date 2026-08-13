"""Cross-validation of the AQRE against pygambit's extensive-form logit solver.

Gambit's predictor–corrector homotopy on the agent normal form is the
authoritative implementation of McKelvey–Palfrey (1998); we inherit its
correctness rather than compete with it, exactly as
:mod:`strataq.core.solve.validate` does for the strategic form.

Optional dependency: install extra ``[gambit]`` (or the dev group). Every
consumer must degrade gracefully when pygambit is absent — the test suite falls
back to hand-derived answers on small trees and says so.

References
----------
Turocy, GEB 2005; McKelvey–Palfrey, Experimental Economics 1998. Tier:
engineering validation.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import jax.numpy as jnp
from jax import Array

from strataq.extensive.tree import CHANCE, TERMINAL, ExtensiveGame

try:  # pragma: no cover - trivially environment-dependent
    import pygambit  # type: ignore[import-untyped]

    HAVE_GAMBIT = True
except ImportError:  # pragma: no cover
    pygambit = None
    HAVE_GAMBIT = False

__all__ = ["HAVE_GAMBIT", "gambit_agent_qre", "max_behaviour_gap", "to_pygambit_tree"]


def _exact(probabilities: Array) -> list[Any]:
    """Chance probabilities as exact rationals — Gambit rejects a float 1/3 + 1/3 + 1/3.

    The last entry absorbs the rounding so the list sums to exactly one.
    """
    fractions = [Fraction(float(p)).limit_denominator(1_000_000) for p in probabilities]
    fractions[-1] = Fraction(1) - sum(fractions[:-1])
    return [pygambit.Rational(f.numerator, f.denominator) for f in fractions]


def to_pygambit_tree(game: ExtensiveGame) -> Any:
    """Convert an :class:`ExtensiveGame` to a pygambit extensive-form game.

    Information sets are rebuilt by attaching later members to the first member's
    Gambit infoset, so the two representations agree about who knows what.
    """
    if not HAVE_GAMBIT:
        raise ImportError("pygambit is not installed; install the [gambit] extra")
    out = pygambit.Game.new_tree(players=list(game.player_labels), title=game.title)
    nodes: dict[int, Any] = {0: out.root}
    first_member: dict[int, Any] = {}
    outcomes: dict[int, Any] = {}

    for node in range(game.n_nodes):
        role = int(game.player[node])
        if role == TERMINAL:
            payoffs = [float(v) for v in game.payoffs[node]]
            outcomes[node] = out.add_outcome(label=f"o{node}", payoffs=payoffs)
            out.set_outcome(nodes[node], outcomes[node])
            continue
        count = int(game.n_actions[node])
        if role == CHANCE:
            out.append_move(nodes[node], out.players.chance, [f"c{a}" for a in range(count)])
            out.set_chance_probs(nodes[node].infoset, _exact(game.chance[node, :count]))
        else:
            h = int(game.infoset[node])
            if h in first_member:
                out.append_infoset(nodes[node], first_member[h].infoset)
            else:
                out.append_move(nodes[node], game.player_labels[role], list(game.action_labels[h]))
                first_member[h] = nodes[node]
        for slot, child in enumerate(list(nodes[node].children)):
            nodes[int(game.children[node, slot])] = child
    return out


def gambit_agent_qre(game: ExtensiveGame, lam: float) -> Array:
    """The AQRE behaviour profile at precision λ, from Gambit's homotopy tracer.

    Returns ``(n_infosets, max_actions)`` in *this* library's information-set
    order, so it lines up with :attr:`~strataq.extensive.aqre.AQREPoint.behaviour`
    directly.
    """
    out = to_pygambit_tree(game)
    solved = pygambit.qre.logit_solve_lambda(out, lam=[float(lam)])[0]
    lookup: dict[tuple[int, int], list[float]] = {}
    index = 0
    for number, player in enumerate(out.players):
        for order, infoset in enumerate(player.infosets):
            count = len(list(infoset.actions))
            lookup[(number, order)] = [float(solved[index + a]) for a in range(count)]
            index += count
    profile = jnp.zeros((game.n_infosets, game.max_actions), dtype=jnp.float64)
    seen: dict[int, int] = {}
    for h in range(game.n_infosets):
        owner = int(game.infoset_player[h])
        order = seen.get(owner, 0)
        seen[owner] = order + 1
        row = lookup[(owner, order)]
        profile = profile.at[h, : len(row)].set(jnp.asarray(row, dtype=jnp.float64))
    return profile


def max_behaviour_gap(ours: Array, theirs: Array) -> float:
    """Sup-norm distance between two behaviour profiles."""
    return float(jnp.max(jnp.abs(jnp.asarray(ours) - jnp.asarray(theirs))))
