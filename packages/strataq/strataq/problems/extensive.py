"""What to do at each decision point in a game that unfolds over time.

A tree is the right model whenever order matters and someone might not see
everything: entry then response, offer then counter-offer, a deal with a hidden
type. The answer is a *behaviour strategy* — what to do at each information set
— and this problem returns two of them: the backward-induction one where the
tree has perfect information, and the agent QRE one, which is what to do when the
other side is not perfectly rational either.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import jax.numpy as jnp
from jax import Array

from strataq.extensive.aqre import AQREPoint, agent_qre
from strataq.extensive.backward import BackwardInduction, backward_induction
from strataq.extensive.behaviour import expected_payoffs
from strataq.extensive.catalogue import CATALOGUE, build
from strataq.extensive.tree import ExtensiveGame, perfect_recall_violations
from strataq.problems.base import (
    Problem,
    Solution,
    Summary,
    check_convergence,
    render,
)

__all__ = ["ExtensiveProblem", "ExtensiveSolution"]


@dataclass(frozen=True, repr=False, eq=False)
class ExtensiveSolution(Solution):
    """The answer to an :class:`ExtensiveProblem`."""

    behaviour: Array
    """``(n_infosets, max_actions)``: the AQRE behaviour strategy."""
    recommended: tuple[str, ...]
    """The most likely action at each information set, by name."""
    infoset_labels: tuple[str, ...]
    infoset_players: tuple[str, ...]
    action_labels: tuple[tuple[str, ...], ...]
    utilities: Array
    """``(n_infosets, max_actions)``: the conditional payoffs behind each choice."""
    expected_payoffs: Array
    """Expected payoff to each player under the AQRE, ``(n_players,)``."""
    reach: Array
    """``(n_nodes,)``: probability of reaching each node under the AQRE."""
    precision: float
    subgame_perfect: Array | None
    """Backward-induction behaviour, or ``None`` for imperfect-information trees."""
    subgame_perfect_payoffs: Array | None
    subgame_perfect_actions: tuple[str, ...]
    """The backward-induction action at each information set; empty when unavailable."""
    perfect_information: bool
    perfect_recall: bool
    n_nodes: int
    n_infosets: int
    players: tuple[str, ...]
    title: str
    success: bool
    message: str
    game: ExtensiveGame = field(repr=False)
    point: AQREPoint = field(repr=False)
    induction: BackwardInduction | None = field(repr=False, default=None)

    @cached_property
    def divergence(self) -> float | None:
        """Sup-norm gap between the AQRE and the backward-induction behaviour.

        The number that says how far quantal play sits from the textbook answer —
        large on the centipede, small on entry deterrence.
        """
        if self.subgame_perfect is None:
            return None
        return float(jnp.max(jnp.abs(self.behaviour - self.subgame_perfect)))

    def summary(self) -> Summary:
        return render(
            f"strataq ExtensiveProblem ({self.title})",
            [
                ("players", len(self.players)),
                ("nodes", self.n_nodes),
                ("information sets", self.n_infosets),
                ("precision", self.precision),
                ("divergence", self.divergence),
            ],
            [
                ("perfect info", self.perfect_information),
                ("perfect recall", self.perfect_recall),
                ("value p0", float(self.expected_payoffs[0])),
                ("converged", self.success),
                ("", None),
            ],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "behaviour": [
                [float(p) for p in row[: len(self.action_labels[h])]]
                for h, row in enumerate(self.behaviour)
            ],
            "recommended": list(self.recommended),
            "infoset_labels": list(self.infoset_labels),
            "infoset_players": list(self.infoset_players),
            "action_labels": [list(names) for names in self.action_labels],
            "utilities": [
                [float(u) for u in row[: len(self.action_labels[h])]]
                for h, row in enumerate(self.utilities)
            ],
            "expected_payoffs": [float(v) for v in self.expected_payoffs],
            "precision": self.precision,
            "subgame_perfect": (
                None
                if self.subgame_perfect is None
                else [
                    [float(p) for p in row[: len(self.action_labels[h])]]
                    for h, row in enumerate(self.subgame_perfect)
                ]
            ),
            "subgame_perfect_payoffs": (
                None
                if self.subgame_perfect_payoffs is None
                else [float(v) for v in self.subgame_perfect_payoffs]
            ),
            "subgame_perfect_actions": list(self.subgame_perfect_actions),
            "divergence": self.divergence,
            "perfect_information": self.perfect_information,
            "perfect_recall": self.perfect_recall,
            "n_nodes": self.n_nodes,
            "n_infosets": self.n_infosets,
            "players": list(self.players),
            "title": self.title,
            "success": self.success,
            "message": self.message,
        }


class ExtensiveProblem(Problem):
    """Agent QRE and backward induction on a game tree.

    Parameters
    ----------
    tree
        An :class:`~strataq.extensive.tree.ExtensiveGame`, the nested dict form,
        or the name of a catalogue tree (``entry_deterrence``, ``centipede``,
        ``bargaining``, ``seltens_horse``, ``kuhn_poker``).
    precision
        Logit precision λ at each information set.
    options
        Keyword arguments for a catalogue constructor, ignored otherwise.
    """

    def __init__(
        self,
        *,
        tree: ExtensiveGame | dict[str, Any] | str,
        precision: float = 1.0,
        options: dict[str, Any] | None = None,
        tol: float | None = None,
        max_iter: int | None = None,
    ) -> None:
        if isinstance(tree, ExtensiveGame):
            game = tree
        elif isinstance(tree, str):
            if tree not in CATALOGUE:
                raise ValueError(f"unknown tree {tree!r}; the catalogue is {sorted(CATALOGUE)}")
            game = build(tree, **(options or {}))
        elif isinstance(tree, dict):
            game = ExtensiveGame.from_dict(tree)
        else:
            raise ValueError("tree must be an ExtensiveGame, a spec dict, or a catalogue name")
        if not float(precision) >= 0:
            raise ValueError(f"precision must be >= 0, got {precision}")
        self.tree = game
        self.precision = float(precision)
        self.tol = tol
        self.max_iter = max_iter

    def solve(self) -> ExtensiveSolution:
        from strataq.core.defaults import base_config

        game = self.tree
        tol = base_config().tolerances.solve if self.tol is None else float(self.tol)
        point = agent_qre(game, self.precision, tol=self.tol, max_iter=self.max_iter)
        success, message = check_convergence(
            bool(point.converged), float(point.residual), tol, "ExtensiveProblem.solve"
        )
        induction: BackwardInduction | None = None
        perfect_actions: tuple[str, ...] = ()
        if game.is_perfect_information:
            induction = backward_induction(game)
            perfect_actions = tuple(
                game.action_labels[h][int(jnp.argmax(induction.behaviour[h]))]
                for h in range(game.n_infosets)
            )
        recommended = tuple(
            game.action_labels[h][int(jnp.argmax(point.behaviour[h]))]
            for h in range(game.n_infosets)
        )
        return ExtensiveSolution(
            behaviour=point.behaviour,
            recommended=recommended,
            infoset_labels=game.infoset_labels,
            infoset_players=tuple(game.player_labels[int(p)] for p in game.infoset_player),
            action_labels=game.action_labels,
            utilities=point.utilities,
            expected_payoffs=point.expected_payoffs,
            reach=point.reach,
            precision=self.precision,
            subgame_perfect=None if induction is None else induction.behaviour,
            subgame_perfect_payoffs=None if induction is None else induction.value,
            subgame_perfect_actions=perfect_actions,
            perfect_information=game.is_perfect_information,
            perfect_recall=not perfect_recall_violations(game),
            n_nodes=game.n_nodes,
            n_infosets=game.n_infosets,
            players=game.player_labels,
            title=game.title,
            success=success,
            message=message,
            game=game,
            point=point,
            induction=induction,
        )


def behaviour_payoffs(game: ExtensiveGame, behaviour: Sequence[Sequence[float]] | Array) -> Array:
    """Expected payoffs of an arbitrary behaviour profile — the evaluation hook."""
    return expected_payoffs(game, jnp.asarray(behaviour, dtype=jnp.float64))
