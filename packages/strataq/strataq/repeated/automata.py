"""Repeated-game strategies as finite automata (Moore machines).

A strategy for one player is a state machine: an action to play in each state
and a transition on the *observed joint action*. Grim trigger is two states,
tit-for-tat is one state per rival action, and anything a user writes is the
same object — which is what makes the incentive machinery in
:mod:`strataq.repeated.folk` generic rather than a special case per strategy.

Automata are combinatorial, not differentiable: the arrays are integer indices
and nothing here is traced.

References
----------
Abreu–Rubinstein, Econometrica 1988 (automaton representation of repeated-game
strategies); Mailath–Samuelson 2006 §2. Tier: exact — this is a definition, not
a result.
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jax import Array

__all__ = ["Automaton", "always", "grim_trigger", "tit_for_tat"]


class Automaton(eqx.Module):
    """A Moore machine playing a repeated game.

    Parameters
    ----------
    actions
        ``(n_states,)`` integer array: the action index played in each state.
    transitions
        ``(n_states,) + num_actions`` integer array: ``transitions[s][a]`` is the
        state entered from ``s`` after the joint action ``a`` is observed.
    initial
        The state the machine starts in.
    """

    actions: Array
    transitions: Array
    initial: int = eqx.field(static=True)

    def __init__(
        self,
        actions: Sequence[int] | Array,
        transitions: Array,
        *,
        initial: int = 0,
    ) -> None:
        acts = jnp.asarray(actions, dtype=jnp.int32).ravel()
        trans = jnp.asarray(transitions, dtype=jnp.int32)
        n_states = int(acts.shape[0])
        if n_states < 1:
            raise ValueError("an automaton needs at least one state")
        if trans.ndim < 2 or int(trans.shape[0]) != n_states:
            raise ValueError(
                f"transitions must have shape (n_states,) + num_actions with n_states="
                f"{n_states}, got {tuple(trans.shape)}"
            )
        if bool(jnp.any(trans < 0)) or bool(jnp.any(trans >= n_states)):
            raise ValueError(f"transitions must land in [0, {n_states})")
        if bool(jnp.any(acts < 0)):
            raise ValueError("actions must be non-negative indices")
        if not 0 <= int(initial) < n_states:
            raise ValueError(f"initial must be in [0, {n_states}), got {initial}")
        self.actions = acts
        self.transitions = trans
        self.initial = int(initial)

    @property
    def n_states(self) -> int:
        """How many states the machine has."""
        return int(self.actions.shape[0])

    @property
    def num_actions(self) -> tuple[int, ...]:
        """The action-count profile of the stage game this machine watches."""
        return tuple(int(m) for m in self.transitions.shape[1:])


def _check_profile(
    num_actions: Sequence[int], profile: Sequence[int], name: str
) -> tuple[int, ...]:
    values = tuple(int(a) for a in profile)
    if len(values) != len(num_actions):
        raise ValueError(f"{name} must have {len(num_actions)} entries, got {len(values)}")
    for i, (a, m) in enumerate(zip(values, num_actions, strict=True)):
        if not 0 <= a < int(m):
            raise ValueError(f"{name}[{i}] must be in [0, {int(m)}), got {a}")
    return values


def always(num_actions: Sequence[int], player: int, action: int) -> Automaton:
    """The one-state machine that plays ``action`` forever."""
    shape = tuple(int(m) for m in num_actions)
    if not 0 <= int(player) < len(shape):
        raise ValueError(f"player must be in [0, {len(shape)}), got {player}")
    if not 0 <= int(action) < shape[player]:
        raise ValueError(f"action must be in [0, {shape[player]}), got {action}")
    return Automaton([int(action)], jnp.zeros((1, *shape), dtype=jnp.int32))


def grim_trigger(
    num_actions: Sequence[int],
    player: int,
    target: Sequence[int],
    punishment: Sequence[int],
) -> Automaton:
    """Play ``target`` while nobody has deviated; play ``punishment`` forever after.

    State 0 is cooperation, state 1 is the absorbing punishment. The trigger is
    the *joint* action: any departure from ``target`` by anyone moves everyone to
    state 1, which is what makes the profile a public-monitoring equilibrium
    candidate.

    References
    ----------
    Friedman, RES 1971 (Nash-reversion folk theorem). Tier: exact.
    """
    shape = tuple(int(m) for m in num_actions)
    if not 0 <= int(player) < len(shape):
        raise ValueError(f"player must be in [0, {len(shape)}), got {player}")
    goal = _check_profile(shape, target, "target")
    punish = _check_profile(shape, punishment, "punishment")
    transitions = jnp.ones((2, *shape), dtype=jnp.int32)
    transitions = transitions.at[(0, *goal)].set(0)
    return Automaton([goal[player], punish[player]], transitions)


def tit_for_tat(
    num_actions: Sequence[int],
    player: int,
    *,
    watch: int | None = None,
    response: Sequence[int] | None = None,
    start: int = 0,
) -> Automaton:
    """Mirror what ``watch`` did last period.

    One state per action of the watched player; ``response[b]`` is what this
    player plays after observing ``b`` (the identity map when both players have
    the same number of actions). ``start`` is the rival action assumed before
    play begins — action 0 by convention, so tit-for-tat opens cooperatively
    when action 0 is cooperation.

    References
    ----------
    Axelrod 1984; Mailath–Samuelson 2006 §2.5. Tier: exact.
    """
    shape = tuple(int(m) for m in num_actions)
    if len(shape) < 2:
        raise ValueError("tit-for-tat needs at least two players")
    if not 0 <= int(player) < len(shape):
        raise ValueError(f"player must be in [0, {len(shape)}), got {player}")
    other = int(watch) if watch is not None else (1 - int(player) if len(shape) == 2 else -1)
    if not 0 <= other < len(shape) or other == int(player):
        raise ValueError("watch must name another player; it is required beyond two players")
    n_states = shape[other]
    if response is None:
        if shape[int(player)] != n_states:
            raise ValueError(
                "response= is required when the players have different action counts "
                f"({shape[int(player)]} vs {n_states})"
            )
        acts = list(range(n_states))
    else:
        acts = [int(a) for a in response]
        if len(acts) != n_states:
            raise ValueError(f"response must have {n_states} entries, got {len(acts)}")
        if any(not 0 <= a < shape[int(player)] for a in acts):
            raise ValueError(f"response entries must be in [0, {shape[int(player)]})")
    if not 0 <= int(start) < n_states:
        raise ValueError(f"start must be in [0, {n_states}), got {start}")
    # transitions[s][a] = a[other]: the state is the rival's last action.
    index = jnp.arange(n_states, dtype=jnp.int32)
    broadcast_shape = [1] * len(shape)
    broadcast_shape[other] = n_states
    rival_last = jnp.broadcast_to(index.reshape(broadcast_shape), shape)
    transitions = jnp.broadcast_to(rival_last[None, ...], (n_states, *shape))
    return Automaton(acts, transitions, initial=int(start))
