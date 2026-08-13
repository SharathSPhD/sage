"""``solve_situation()`` — the whole recommendation, assembled once, server side.

Every client that shows a user "here is what to do" needs the same five things:
the action, what the other side is likely to do, what the runners-up would have
paid, how confident the call is, and whether it survives a different assumption
about how sharp everyone is. Assembling those from a raw equilibrium is exactly
the kind of logic that drifts apart between a notebook, a service and a browser.

So it is assembled here, once. A :class:`Situation` is a payoff table, whose
player you are, and a precision; :meth:`Situation.solve` returns a
:class:`SituationSolution` that a caller renders and does not compute from.

The sensitivity block is the part worth reading: the recommendation is re-solved
across a ladder of precisions around the one you stated, and the solution reports
the fraction of the ladder that agrees and the precision at which the answer
first changes. A recommendation that holds from λ/4 to 4λ is a different object
from one that holds only at the λ you happened to pick.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import jax.numpy as jnp
from jax import Array

from strataq.core.solve.fixedpoint import logit_qre
from strataq.core.types import QREPoint
from strataq.finite.games.tensor import DenseTensorGame
from strataq.problems.base import (
    Diagnostics,
    Problem,
    Solution,
    Summary,
    finite_diagnostics,
    render,
)

__all__ = [
    "Alternative",
    "RivalView",
    "Sensitivity",
    "Situation",
    "SituationSolution",
    "solve_situation",
]

MAX_PROFILES = 250_000


@dataclass(frozen=True)
class Alternative:
    """One of your actions, scored against the equilibrium rival play."""

    action: int
    label: str
    expected_payoff: float
    regret: float
    """How much worse than the recommendation — zero for the recommendation itself."""
    probability: float
    """Your own equilibrium probability of playing it."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "label": self.label,
            "expected_payoff": self.expected_payoff,
            "regret": self.regret,
            "probability": self.probability,
        }


@dataclass(frozen=True)
class RivalView:
    """What one other party is expected to do."""

    player: int
    label: str
    actions: tuple[str, ...]
    distribution: tuple[float, ...]
    most_likely: str
    entropy: float
    """Nats. Zero means a confident prediction; log(n) means no information."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "player": self.player,
            "label": self.label,
            "actions": list(self.actions),
            "distribution": list(self.distribution),
            "most_likely": self.most_likely,
            "entropy": self.entropy,
        }


@dataclass(frozen=True)
class Sensitivity:
    """Does the recommendation survive a different guess about how sharp play is?"""

    precisions: tuple[float, ...]
    recommended: tuple[int, ...]
    """The recommended action at each precision on the ladder."""
    payoffs: tuple[float, ...]
    robustness: float
    """Fraction of the ladder that recommends the same action as the stated precision."""
    switch_precision: float | None
    """The lowest precision at which the recommendation differs; ``None`` if never."""
    stable: bool
    """True when the whole ladder agrees."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "precisions": list(self.precisions),
            "recommended": list(self.recommended),
            "payoffs": list(self.payoffs),
            "robustness": self.robustness,
            "switch_precision": self.switch_precision,
            "stable": self.stable,
        }


@dataclass(frozen=True, repr=False, eq=False)
class SituationSolution(Solution):
    """A complete recommendation: action, rivals, alternatives, sensitivity."""

    action: int
    action_label: str
    expected_payoff: float
    confidence: float
    """Gap to the runner-up, as a fraction of the game's payoff range."""
    own_distribution: Array
    """Your own equilibrium mix, ``(n_actions,)`` — the recommendation is its argmax payoff."""
    alternatives: tuple[Alternative, ...]
    """Every action of yours, best first."""
    rivals: tuple[RivalView, ...]
    sensitivity: Sensitivity
    you: int
    n_players: int
    precision: float
    actions: tuple[str, ...]
    players: tuple[str, ...]
    success: bool
    message: str
    game: DenseTensorGame = field(repr=False)
    point: QREPoint = field(repr=False)

    @cached_property
    def diagnostics(self) -> Diagnostics:
        """α, ℛ, ρ(SB), σ_EP — computed on first access, never on the solve path."""
        return finite_diagnostics(self.game, self.point)

    @property
    def runner_up(self) -> Alternative | None:
        """The best alternative to the recommendation, if there is one."""
        return self.alternatives[1] if len(self.alternatives) > 1 else None

    def summary(self) -> Summary:
        rival = self.rivals[0] if self.rivals else None
        return render(
            "strataq Situation",
            [
                ("action", self.action_label),
                ("expected payoff", self.expected_payoff),
                ("confidence", self.confidence),
                ("robustness", self.sensitivity.robustness),
                ("switches at", self.sensitivity.switch_precision),
            ],
            [
                ("you", self.players[self.you]),
                ("players", self.n_players),
                ("your actions", len(self.actions)),
                ("precision", self.precision),
                ("rival expects", None if rival is None else rival.most_likely),
            ],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "action_label": self.action_label,
            "expected_payoff": self.expected_payoff,
            "confidence": self.confidence,
            "own_distribution": [float(p) for p in self.own_distribution],
            "alternatives": [a.as_dict() for a in self.alternatives],
            "rivals": [r.as_dict() for r in self.rivals],
            "sensitivity": self.sensitivity.as_dict(),
            "you": self.you,
            "n_players": self.n_players,
            "precision": self.precision,
            "actions": list(self.actions),
            "players": list(self.players),
            "success": self.success,
            "message": self.message,
        }


class Situation(Problem):
    """A strategic situation stated in the caller's own words.

    Parameters
    ----------
    payoffs
        One payoff tensor per party, or a :class:`DenseTensorGame`.
    you
        Which party the recommendation is for.
    precision
        Logit precision λ — how sharply everyone, including you, is assumed to
        respond to payoff differences.
    actions
        Names for *your* actions; ``rival_actions`` names theirs.
    players
        Names for the parties.
    ladder
        Explicit precisions for the sensitivity sweep; defaults to the geometric
        ladder around ``precision`` in ``config/base.yaml``.
    """

    def __init__(
        self,
        *,
        payoffs: DenseTensorGame | Sequence[Any],
        you: int = 0,
        precision: float = 1.0,
        actions: Sequence[str] | None = None,
        rival_actions: Sequence[Sequence[str]] | None = None,
        players: Sequence[str] | None = None,
        ladder: Sequence[float] | None = None,
        tol: float | None = None,
        max_iter: int | None = None,
    ) -> None:
        game = (
            payoffs
            if isinstance(payoffs, DenseTensorGame)
            else DenseTensorGame(tuple(jnp.asarray(u, dtype=jnp.float64) for u in payoffs))
        )
        if not all(bool(jnp.all(jnp.isfinite(u))) for u in game.payoffs):
            raise ValueError("payoffs must be finite")
        size = 1
        for m in game.num_actions:
            size *= m
        if size > MAX_PROFILES:
            raise ValueError(
                f"Situation: {size} joint profiles exceeds the dense limit {MAX_PROFILES}"
            )
        if not 0 <= int(you) < game.n_players:
            raise ValueError(f"you must be in [0, {game.n_players}), got {you}")
        if not float(precision) > 0:
            raise ValueError(f"precision must be > 0, got {precision}")
        self.game_spec = game
        self.you = int(you)
        self.precision = float(precision)
        self.player_labels = _labels(players, game.n_players, "P")
        self.action_labels = _action_labels(game, self.you, actions, rival_actions)
        self.ladder = None if ladder is None else tuple(float(v) for v in ladder)
        self.tol = tol
        self.max_iter = max_iter

    def _recommend(self, precision: float) -> tuple[QREPoint, int, float]:
        point = logit_qre(self.game_spec, precision, tol=self.tol, max_iter=self.max_iter)
        curve = point.expected_payoffs[self.you]
        best = int(jnp.argmax(curve))
        return point, best, float(curve[best])

    def _ladder(self) -> tuple[float, ...]:
        if self.ladder is not None:
            return self.ladder
        cfg = base_ladder()
        return tuple(
            float(v)
            for v in jnp.geomspace(self.precision * cfg[1], self.precision * cfg[2], int(cfg[0]))
        )

    def solve(self) -> SituationSolution:
        point, best, payoff = self._recommend(self.precision)
        curve = point.expected_payoffs[self.you]
        own = point.sigma[self.you]
        scale = float(self.game_spec.payoff_range)
        order = [int(i) for i in jnp.argsort(-curve)]
        alternatives = tuple(
            Alternative(
                action=i,
                label=self.action_labels[self.you][i],
                expected_payoff=float(curve[i]),
                regret=float(curve[best] - curve[i]),
                probability=float(own[i]),
            )
            for i in order
        )
        second = alternatives[1].expected_payoff if len(alternatives) > 1 else payoff
        confidence = float((payoff - second) / scale) if scale > 0 else 0.0

        rivals = []
        for i in range(self.game_spec.n_players):
            if i == self.you:
                continue
            mix = point.sigma[i]
            names = self.action_labels[i]
            entropy = float(
                -jnp.sum(
                    jnp.where(mix > 0, mix * jnp.log(jnp.maximum(mix, sys.float_info.min)), 0.0)
                )
            )
            rivals.append(
                RivalView(
                    player=i,
                    label=self.player_labels[i],
                    actions=names,
                    distribution=tuple(float(p) for p in mix),
                    most_likely=names[int(jnp.argmax(mix))],
                    entropy=entropy,
                )
            )

        precisions = self._ladder()
        picks = []
        payoffs = []
        for value in precisions:
            _, pick, gain = self._recommend(value)
            picks.append(pick)
            payoffs.append(gain)
        agree = [p == best for p in picks]
        switch = next((float(v) for v, ok in zip(precisions, agree, strict=True) if not ok), None)
        sensitivity = Sensitivity(
            precisions=tuple(precisions),
            recommended=tuple(picks),
            payoffs=tuple(payoffs),
            robustness=float(sum(agree) / len(agree)),
            switch_precision=switch,
            stable=all(agree),
        )
        return SituationSolution(
            action=best,
            action_label=self.action_labels[self.you][best],
            expected_payoff=payoff,
            confidence=confidence,
            own_distribution=own,
            alternatives=alternatives,
            rivals=tuple(rivals),
            sensitivity=sensitivity,
            you=self.you,
            n_players=self.game_spec.n_players,
            precision=self.precision,
            actions=self.action_labels[self.you],
            players=self.player_labels,
            success=bool(point.converged),
            message="converged" if bool(point.converged) else "solve did not reach tolerance",
            game=self.game_spec,
            point=point,
        )


def base_ladder() -> tuple[int, float, float]:
    """``(points, low multiple, high multiple)`` for the sensitivity sweep."""
    from strataq.core.defaults import base_config

    cfg = base_config().situation
    return int(cfg.ladder_points), float(cfg.ladder_low), float(cfg.ladder_high)


def _labels(names: Sequence[str] | None, count: int, prefix: str) -> tuple[str, ...]:
    if names is None:
        return tuple(f"{prefix}{i}" for i in range(count))
    values = tuple(str(n) for n in names)
    if len(values) != count:
        raise ValueError(f"expected {count} names, got {len(values)}")
    return values


def _action_labels(
    game: DenseTensorGame,
    you: int,
    actions: Sequence[str] | None,
    rival_actions: Sequence[Sequence[str]] | None,
) -> tuple[tuple[str, ...], ...]:
    out: list[tuple[str, ...]] = []
    rivals = list(rival_actions) if rival_actions is not None else None
    slot = 0
    for i, count in enumerate(game.num_actions):
        if i == you:
            out.append(_labels(actions, count, "a"))
        elif rivals is not None:
            if slot >= len(rivals):
                raise ValueError(f"rival_actions needs {game.n_players - 1} lists")
            out.append(_labels(rivals[slot], count, "a"))
            slot += 1
        else:
            out.append(_labels(None, count, "a"))
    return tuple(out)


def solve_situation(
    payoffs: DenseTensorGame | Sequence[Any],
    *,
    you: int = 0,
    precision: float = 1.0,
    actions: Sequence[str] | None = None,
    rival_actions: Sequence[Sequence[str]] | None = None,
    players: Sequence[str] | None = None,
    ladder: Sequence[float] | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
) -> SituationSolution:
    """State a situation, get the whole recommendation.

    The one-call form of :class:`Situation`. Returns the recommended action, what
    each other party is expected to do, every alternative with its regret, and
    whether the answer survives a different assumption about precision — so no
    caller ever assembles a recommendation from an equilibrium.

    Examples
    --------
    >>> import strataq as sq
    >>> res = sq.solve_situation(
    ...     [[[3.0, 0.0], [5.0, 1.0]], [[3.0, 5.0], [0.0, 1.0]]],
    ...     actions=["cooperate", "defect"],
    ...     precision=2.0,
    ... )
    >>> res.action_label
    'defect'
    """
    return Situation(
        payoffs=payoffs,
        you=you,
        precision=precision,
        actions=actions,
        rival_actions=rival_actions,
        players=players,
        ladder=ladder,
        tol=tol,
        max_iter=max_iter,
    ).solve()
