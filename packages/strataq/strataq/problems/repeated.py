"""Will the deal hold? Sustainability of an agreement that repeats.

A stage game plus a discount factor is the standard model of an ongoing
relationship — a cartel, a supply contract, a tacit understanding between two
firms. The question is never "what is the equilibrium" (there are too many) but
"is *this* arrangement self-enforcing, and how patient must the parties be".

``RepeatedProblem`` answers both: the critical discount factor for the profile
you name, and the whole set of pure profiles grim trigger sustains at the
discount factor you have.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import jax.numpy as jnp
from jax import Array

from strataq.finite.games.tensor import DenseTensorGame
from strataq.problems.base import (
    Problem,
    Solution,
    Summary,
    render,
)
from strataq.repeated.automata import Automaton, grim_trigger
from strataq.repeated.folk import (
    RepeatedLogitPoint,
    SustainableSet,
    best_deviation,
    critical_discount,
    deviation_gains,
    grim_critical_discount,
    logit_trigger_equilibrium,
    minmax_payoffs,
    profile_payoffs,
    sustainable_payoff_set,
)

__all__ = ["RepeatedProblem", "RepeatedSolution"]

MAX_PROFILES = 100_000


@dataclass(frozen=True, repr=False, eq=False)
class RepeatedSolution(Solution):
    """The answer to a :class:`RepeatedProblem`."""

    sustainable: bool
    """Does grim trigger sustain ``target`` at this discount factor?"""
    critical_discount: float
    """The smallest δ that sustains ``target`` — ``inf`` if no δ does."""
    critical_by_player: Array
    """Per-player critical δ, ``(n_players,)``: whoever is largest binds."""
    target: Array
    """The profile being sustained, ``(n_players,)`` action indices."""
    target_payoffs: Array
    """Stage payoffs at ``target``, ``(n_players,)``."""
    deviation_payoffs: Array
    """Best one-shot deviation payoff per player, ``(n_players,)``."""
    punishment_payoffs: Array
    """The per-period payoff after a deviation, ``(n_players,)``."""
    discount: float
    sustainable_profiles: Array
    """Every pure profile grim trigger sustains at ``discount``, ``(k, n_players)``."""
    sustainable_payoffs: Array
    """Their payoff vectors, ``(k, n_players)``."""
    frontier: Array
    """Pareto-undominated sustainable payoff vectors, ``(m, n_players)``."""
    all_critical: Array
    """Critical δ for every pure profile, ``(n_profiles,)``."""
    cooperation_probability: float | None
    """P(``target`` is played) in the logit trigger equilibrium; ``None`` unless asked."""
    precision: float | None
    success: bool
    message: str
    game: DenseTensorGame = field(repr=False)
    logit_point: RepeatedLogitPoint | None = field(repr=False, default=None)
    sustainable_set: SustainableSet | None = field(repr=False, default=None)

    @cached_property
    def gain_from_deviating(self) -> Array:
        """``d_i - u_i(target)``: what a one-shot deviation is worth, per player."""
        return self.deviation_payoffs - self.target_payoffs

    def summary(self) -> Summary:
        binding = int(jnp.argmax(self.critical_by_player))
        return render(
            "strataq RepeatedProblem",
            [
                ("sustainable", self.sustainable),
                ("critical delta", self.critical_discount),
                ("delta", self.discount),
                ("binding player", binding),
                ("cooperation prob", self.cooperation_probability),
            ],
            [
                ("players", int(self.target.shape[0])),
                ("profiles", int(self.all_critical.shape[0])),
                ("sustainable", int(self.sustainable_profiles.shape[0])),
                ("frontier", int(self.frontier.shape[0])),
                ("precision", self.precision),
            ],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "sustainable": self.sustainable,
            "critical_discount": self.critical_discount,
            "critical_by_player": [float(d) for d in self.critical_by_player],
            "target": [int(a) for a in self.target],
            "target_payoffs": [float(u) for u in self.target_payoffs],
            "deviation_payoffs": [float(u) for u in self.deviation_payoffs],
            "punishment_payoffs": [float(u) for u in self.punishment_payoffs],
            "discount": self.discount,
            "sustainable_profiles": [[int(a) for a in row] for row in self.sustainable_profiles],
            "sustainable_payoffs": [[float(u) for u in row] for row in self.sustainable_payoffs],
            "frontier": [[float(u) for u in row] for row in self.frontier],
            "all_critical": [float(d) for d in self.all_critical],
            "cooperation_probability": self.cooperation_probability,
            "precision": self.precision,
            "success": self.success,
            "message": self.message,
        }


class RepeatedProblem(Problem):
    """Sustainability of a repeated agreement.

    Parameters
    ----------
    payoffs
        One payoff tensor per player, or a :class:`DenseTensorGame`.
    discount
        The per-period discount factor δ ∈ [0, 1).
    target
        The pure profile to sustain. Defaults to the profile maximising the sum
        of payoffs — the cooperative outcome an agreement would be about.
    punishment
        Per-player punishment payoffs, or ``"minmax"`` (the default, pure-strategy
        minmax) or ``"nash"`` to punish by reverting to the worst pure Nash
        equilibrium of the stage game.
    precision
        Optional logit precision λ. Supplying it also solves the logit trigger
        equilibrium and reports how *probable* cooperation is, rather than only
        whether it is incentive compatible.
    """

    def __init__(
        self,
        *,
        payoffs: DenseTensorGame | Sequence[Any],
        discount: float,
        target: Sequence[int] | None = None,
        punishment: str | Sequence[float] = "minmax",
        precision: float | None = None,
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
                f"RepeatedProblem: {size} joint profiles exceeds the dense limit "
                f"{MAX_PROFILES}; coarsen the stage game."
            )
        if not 0.0 <= float(discount) < 1.0:
            raise ValueError(f"discount must be in [0, 1), got {discount}")
        if precision is not None and not float(precision) >= 0:
            raise ValueError(f"precision must be >= 0, got {precision}")
        self.game_spec = game
        self.discount = float(discount)
        self.punishment = self._resolve_punishment(game, punishment)
        self.target = self._resolve_target(game, target)
        self.precision = None if precision is None else float(precision)
        self.tol = tol
        self.max_iter = max_iter

    @staticmethod
    def _resolve_punishment(game: DenseTensorGame, punishment: str | Sequence[float]) -> Array:
        if isinstance(punishment, str):
            if punishment == "minmax":
                return minmax_payoffs(game)
            if punishment == "nash":
                return _worst_pure_nash(game)
            raise ValueError(
                f"punishment must be 'minmax', 'nash' or a payoff vector, got {punishment!r}"
            )
        values = jnp.asarray(punishment, dtype=jnp.float64).ravel()
        if values.shape != (game.n_players,):
            raise ValueError(
                f"punishment must have {game.n_players} entries, got {int(values.shape[0])}"
            )
        return values

    @staticmethod
    def _resolve_target(game: DenseTensorGame, target: Sequence[int] | None) -> tuple[int, ...]:
        if target is None:
            total = jnp.sum(jnp.stack(game.payoffs), axis=0)
            flat = int(jnp.argmax(total.ravel()))
            return tuple(int(a) for a in jnp.unravel_index(flat, game.num_actions))
        chosen = tuple(int(a) for a in target)
        if len(chosen) != game.n_players:
            raise ValueError(f"target must have {game.n_players} entries, got {len(chosen)}")
        for i, (a, m) in enumerate(zip(chosen, game.num_actions, strict=True)):
            if not 0 <= a < m:
                raise ValueError(f"target[{i}] must be in [0, {m}), got {a}")
        return chosen

    def automata(self) -> tuple[Automaton, ...]:
        """The grim-trigger machines that sustain ``target`` against ``punishment``."""
        worst = _punishment_profile(self.game_spec, self.punishment)
        return tuple(
            grim_trigger(self.game_spec.num_actions, i, self.target, worst)
            for i in range(self.game_spec.n_players)
        )

    def solve(self) -> RepeatedSolution:
        game = self.game_spec
        per_player = grim_critical_discount(game, self.target, punishment=self.punishment)
        critical = float(jnp.max(per_player))
        sustainable = bool(critical <= self.discount)
        found = sustainable_payoff_set(game, self.discount, punishment=self.punishment)

        cooperation: float | None = None
        point: RepeatedLogitPoint | None = None
        if self.precision is not None:
            machines = self.automata()
            point = logit_trigger_equilibrium(
                game,
                machines,
                self.discount,
                self.precision,
                tol=self.tol,
                max_iter=self.max_iter,
            )
            start = int(point.initial)
            cooperation = float(
                jnp.prod(
                    jnp.stack(
                        [point.sigma[i][start, self.target[i]] for i in range(game.n_players)]
                    )
                )
            )

        message = (
            f"grim trigger sustains the target for delta >= {critical:.4g}"
            if jnp.isfinite(critical)
            else "the target cannot be sustained at any discount factor"
        )
        return RepeatedSolution(
            sustainable=sustainable,
            critical_discount=critical,
            critical_by_player=per_player,
            target=jnp.asarray(self.target, dtype=jnp.int32),
            target_payoffs=profile_payoffs(game, self.target),
            deviation_payoffs=best_deviation(game, self.target),
            punishment_payoffs=self.punishment,
            discount=self.discount,
            sustainable_profiles=found.sustainable_profiles,
            sustainable_payoffs=found.sustainable_payoffs,
            frontier=found.frontier,
            all_critical=found.critical,
            cooperation_probability=cooperation,
            precision=self.precision,
            success=True,
            message=message,
            game=game,
            logit_point=point,
            sustainable_set=found,
        )


def _pure_nash(game: DenseTensorGame) -> list[tuple[int, ...]]:
    """Every pure-strategy Nash equilibrium of the stage game."""
    import itertools

    found = []
    for profile in itertools.product(*(range(m) for m in game.num_actions)):
        deviation = best_deviation(game, profile)
        payoff = profile_payoffs(game, profile)
        if bool(jnp.all(payoff >= deviation - 0.0)):
            found.append(profile)
    return found


def _worst_pure_nash(game: DenseTensorGame) -> Array:
    """Payoffs of the pure Nash equilibrium with the smallest payoff sum."""
    equilibria = _pure_nash(game)
    if not equilibria:
        raise ValueError(
            "punishment='nash' needs a pure-strategy Nash equilibrium of the stage game; "
            "this game has none. Use 'minmax' or supply punishment payoffs."
        )
    payoffs = [profile_payoffs(game, p) for p in equilibria]
    worst = int(jnp.argmin(jnp.stack([jnp.sum(u) for u in payoffs])))
    return payoffs[worst]


def _punishment_profile(game: DenseTensorGame, punishment: Array) -> tuple[int, ...]:
    """A pure profile delivering (weakly) the punishment payoffs, for the machine.

    The machine has to *play* something in the punishment state. The closest pure
    profile is the one minimising the squared gap to the requested payoffs; for
    minmax and Nash punishment that is the punishing profile itself.
    """
    import itertools

    best: tuple[int, ...] | None = None
    best_gap = float("inf")
    for profile in itertools.product(*(range(m) for m in game.num_actions)):
        gap = jnp.sum((profile_payoffs(game, profile) - punishment) ** 2)
        if float(gap) < best_gap:
            best_gap = float(gap)
            best = profile
    assert best is not None
    return best


def deviation_report(
    game: DenseTensorGame, automata: Sequence[Automaton], discount: float
) -> dict[str, Any]:
    """One-shot deviation gains and the bisected critical δ for any machine profile.

    The escape hatch from grim trigger: tit-for-tat, limited punishment, or a
    machine you wrote get the same treatment, just without the closed form.
    """
    gains = deviation_gains(game, automata, discount)
    return {
        "deviation_gains": [float(g) for g in gains],
        "sustainable": bool(jnp.max(gains) <= 0.0),
        "critical_discount": critical_discount(game, automata),
    }
