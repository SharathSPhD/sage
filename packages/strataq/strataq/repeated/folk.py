"""Incentive compatibility in the infinitely repeated game.

Three layers, cheapest first.

*Closed form.* For grim trigger sustaining a pure profile ``a*`` with punishment
payoffs ``p``, player ``i`` deviates iff ``u_i(a*) < (1-δ) d_i + δ p_i`` where
``d_i`` is its best one-shot deviation, so the critical discount factor is
``δ_i = (d_i - u_i(a*)) / (d_i - p_i)`` and the profile is sustainable iff
``δ ≥ max_i δ_i``. That is the folk-theorem arithmetic and nothing more.

*Generic.* For an arbitrary profile of automata the same question is the
one-shot-deviation criterion: values solve ``V = u + δ P V`` on the joint
machine state, and the profile is subgame perfect iff no player gains by
deviating once in any state and reverting. This covers tit-for-tat, limited
punishment, and anything a user builds.

*Quantal.* :func:`logit_trigger_equilibrium` replaces "deviate iff profitable"
with a logit choice over continuation values at every machine state — the
repeated-game analogue of the logit QRE, and the object that makes
sustainability a probability rather than a bit.

References
----------
Friedman, RES 1971; Abreu–Rubinstein, Econometrica 1988; Fudenberg–Maskin,
Econometrica 1986 (folk theorem). One-shot deviation principle: Blackwell 1965.
Tier: exact. The quantal layer is McKelvey–Palfrey GEB 1995 applied to the
automaton-state game; tier: derived.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from strataq.core.defaults import base_config
from strataq.finite.games.tensor import DenseTensorGame
from strataq.repeated.automata import Automaton

__all__ = [
    "MachineValues",
    "RepeatedLogitPoint",
    "SustainableSet",
    "best_deviation",
    "critical_discount",
    "deviation_gains",
    "grim_critical_discount",
    "is_sustainable",
    "logit_trigger_equilibrium",
    "machine_values",
    "minmax_payoffs",
    "reachable_states",
    "sustainable_payoff_set",
]


def _check_delta(delta: float) -> float:
    value = float(delta)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"discount factor must be in [0, 1), got {value}")
    return value


def _joint_profiles(num_actions: tuple[int, ...]) -> list[tuple[int, ...]]:
    return list(itertools.product(*(range(m) for m in num_actions)))


def minmax_payoffs(game: DenseTensorGame) -> Array:
    """``p_i = min_{a_{-i}} max_{a_i} u_i`` — the pure-strategy minmax, per player.

    Rivals may not randomise here, so this is an upper bound on the mixed minmax
    and therefore a *conservative* punishment: critical discount factors computed
    against it are no smaller than the true ones.
    """
    out = []
    for i in range(game.n_players):
        u = game.payoff_tensor(i)
        best = jnp.max(u, axis=i)  # max over own action, per rival profile
        out.append(jnp.min(best))
    return jnp.stack(out)


def best_deviation(game: DenseTensorGame, profile: Sequence[int]) -> Array:
    """``d_i = max_{a_i} u_i(a_i, a_{-i})`` at the given pure profile, per player."""
    target = tuple(int(a) for a in profile)
    if len(target) != game.n_players:
        raise ValueError(f"profile must have {game.n_players} entries, got {len(target)}")
    for i, (a, m) in enumerate(zip(target, game.num_actions, strict=True)):
        if not 0 <= a < m:
            raise ValueError(f"profile[{i}] must be in [0, {m}), got {a}")
    out = []
    for i in range(game.n_players):
        u = game.payoff_tensor(i)
        rivals = tuple(a for j, a in enumerate(target) if j != i)
        line = jnp.moveaxis(u, i, 0)[(slice(None), *rivals)]
        out.append(jnp.max(line))
    return jnp.stack(out)


def profile_payoffs(game: DenseTensorGame, profile: Sequence[int]) -> Array:
    """``u_i(a)`` at a pure profile, per player."""
    target = tuple(int(a) for a in profile)
    return jnp.stack([game.payoff_tensor(i)[target] for i in range(game.n_players)])


def grim_critical_discount(
    game: DenseTensorGame,
    profile: Sequence[int],
    *,
    punishment: Sequence[float] | Array | None = None,
) -> Array:
    """Per-player critical δ for grim trigger, ``(d_i - u_i) / (d_i - p_i)``.

    Zero where the profile is already a stage best response; ``inf`` where the
    deviation payoff cannot be punished (``d_i <= p_i`` but ``u_i < d_i``), which
    marks the profile as unsustainable at any discount factor.
    """
    punish = (
        minmax_payoffs(game)
        if punishment is None
        else jnp.asarray(punishment, dtype=jnp.float64).ravel()
    )
    if punish.shape != (game.n_players,):
        raise ValueError(f"punishment must have {game.n_players} entries, got {punish.shape}")
    gain = best_deviation(game, profile) - profile_payoffs(game, profile)
    spread = best_deviation(game, profile) - punish
    safe = jnp.where(spread > 0, spread, 1.0)
    critical = jnp.where(gain <= 0.0, 0.0, jnp.where(spread > 0, gain / safe, jnp.inf))
    return jnp.clip(critical, 0.0, jnp.inf)


class SustainableSet(eqx.Module):
    """Which pure profiles grim trigger sustains at a given discount factor."""

    profiles: Array
    """Every pure profile of the stage game, ``(n_profiles, n_players)`` integers."""
    payoffs: Array
    """Stage payoffs at each profile, ``(n_profiles, n_players)``."""
    critical: Array
    """Critical δ for each profile, ``(n_profiles,)`` — ``inf`` where unsustainable."""
    sustainable: Array
    """Boolean mask, ``(n_profiles,)``: ``critical <= delta``."""
    individually_rational: Array
    """Boolean mask: every player weakly above its punishment payoff."""
    punishment: Array
    """The punishment payoffs used, ``(n_players,)``."""
    delta: Array
    """The discount factor the mask was evaluated at."""

    @property
    def sustainable_payoffs(self) -> Array:
        """The sustainable payoff vectors, ``(k, n_players)``."""
        return self.payoffs[self.sustainable]

    @property
    def sustainable_profiles(self) -> Array:
        """The sustainable pure profiles, ``(k, n_players)``."""
        return self.profiles[self.sustainable]

    @property
    def frontier(self) -> Array:
        """Pareto-undominated sustainable payoff vectors, ``(m, n_players)``."""
        points = self.sustainable_payoffs
        if int(points.shape[0]) == 0:
            return points
        dominated = jnp.any(
            jnp.all(points[None, :, :] >= points[:, None, :], axis=-1)
            & jnp.any(points[None, :, :] > points[:, None, :], axis=-1),
            axis=1,
        )
        return points[~dominated]


def sustainable_payoff_set(
    game: DenseTensorGame,
    delta: float,
    *,
    punishment: Sequence[float] | Array | None = None,
) -> SustainableSet:
    """The grim-trigger sustainable set: every pure profile, scored by critical δ.

    The set is over *pure stage profiles*, not the convex hull: public
    randomisation would fill in the hull of :attr:`SustainableSet.frontier`, and
    reporting the vertices is the honest primitive.
    """
    value = _check_delta(delta)
    punish = (
        minmax_payoffs(game)
        if punishment is None
        else jnp.asarray(punishment, dtype=jnp.float64).ravel()
    )
    profiles = _joint_profiles(game.num_actions)
    payoffs = jnp.stack([profile_payoffs(game, p) for p in profiles])
    critical = jnp.stack(
        [jnp.max(grim_critical_discount(game, p, punishment=punish)) for p in profiles]
    )
    return SustainableSet(
        profiles=jnp.asarray(profiles, dtype=jnp.int32),
        payoffs=payoffs,
        critical=critical,
        sustainable=critical <= value,
        individually_rational=jnp.all(payoffs >= punish[None, :], axis=1),
        punishment=punish,
        delta=jnp.asarray(value),
    )


class MachineValues(eqx.Module):
    """Discounted values of an automaton profile, state by state."""

    states: Array
    """The joint machine states, ``(n_states, n_players)`` integers."""
    values: Array
    """``V_i(s)``, the discounted sum from state ``s``, ``(n_states, n_players)``."""
    actions: Array
    """The stage profile played in each state, ``(n_states, n_players)`` integers."""
    successor: Array
    """The state entered next, ``(n_states,)``."""
    initial: Array
    """Index of the state the profile starts in."""
    delta: Array

    @property
    def average_values(self) -> Array:
        """``(1-δ) V``: per-period payoffs, comparable with stage payoffs."""
        return (1.0 - self.delta) * self.values

    @property
    def path_value(self) -> Array:
        """Per-period payoff along the path actually played, ``(n_players,)``."""
        return self.average_values[self.initial]


def _machine_tables(
    game: DenseTensorGame, automata: Sequence[Automaton]
) -> tuple[list[tuple[int, ...]], dict[tuple[int, ...], int], list[tuple[int, ...]]]:
    if len(automata) != game.n_players:
        raise ValueError(f"need one automaton per player ({game.n_players}), got {len(automata)}")
    for i, machine in enumerate(automata):
        if machine.num_actions != game.num_actions:
            raise ValueError(
                f"automaton {i} watches action counts {machine.num_actions}, "
                f"stage game has {game.num_actions}"
            )
    states = list(itertools.product(*(range(m.n_states) for m in automata)))
    index = {s: k for k, s in enumerate(states)}
    plays = [tuple(int(m.actions[s_i]) for m, s_i in zip(automata, s, strict=True)) for s in states]
    return states, index, plays


def _successor(
    automata: Sequence[Automaton], state: tuple[int, ...], action: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(int(m.transitions[(s_i, *action)]) for m, s_i in zip(automata, state, strict=True))


def machine_values(
    game: DenseTensorGame, automata: Sequence[Automaton], delta: float
) -> MachineValues:
    """Solve ``V = u + δ P V`` on the joint machine state space.

    The dynamics are deterministic, so ``P`` is a 0/1 successor matrix and the
    solve is exact — no simulation, no truncation of the infinite horizon.
    """
    value = _check_delta(delta)
    states, index, plays = _machine_tables(game, automata)
    n = len(states)
    stage = jnp.stack([profile_payoffs(game, a) for a in plays])
    successors = [index[_successor(automata, s, a)] for s, a in zip(states, plays, strict=True)]
    transition = jnp.zeros((n, n), dtype=jnp.float64)
    transition = transition.at[jnp.arange(n), jnp.asarray(successors)].set(1.0)
    values = jnp.linalg.solve(jnp.eye(n) - value * transition, stage)
    start = index[tuple(m.initial for m in automata)]
    return MachineValues(
        states=jnp.asarray(states, dtype=jnp.int32),
        values=values,
        actions=jnp.asarray(plays, dtype=jnp.int32),
        successor=jnp.asarray(successors, dtype=jnp.int32),
        initial=jnp.asarray(start, dtype=jnp.int32),
        delta=jnp.asarray(value),
    )


def reachable_states(
    game: DenseTensorGame, automata: Sequence[Automaton]
) -> tuple[tuple[int, ...], ...]:
    """Machine states any history can actually produce, from the initial state.

    The product of the players' state spaces contains combinations no play path
    reaches — one player still cooperating while another has already switched to
    punishment, say. Checking incentives there is not conservatism, it is a
    different game, so the closure under *every* joint action from the initial
    state is what the deviation check uses.
    """
    _, index, _ = _machine_tables(game, automata)
    start = tuple(m.initial for m in automata)
    frontier = [start]
    seen = {start}
    joint = _joint_profiles(game.num_actions)
    while frontier:
        state = frontier.pop()
        for action in joint:
            nxt = _successor(automata, state, action)
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    del index
    return tuple(sorted(seen))


def deviation_gains(game: DenseTensorGame, automata: Sequence[Automaton], delta: float) -> Array:
    """Largest one-shot deviation gain per player, over every reachable machine state.

    Non-positive for every player is exactly the one-shot-deviation criterion:
    the profile is then subgame perfect on the histories play can reach. That is
    subgame perfection in the repeated game itself whenever the machine's states
    cover every reachable history, which grim trigger and tit-for-tat do.
    """
    solved = machine_values(game, automata, delta)
    _, index, _ = _machine_tables(game, automata)
    live = reachable_states(game, automata)
    plays_by_state = {
        state: tuple(int(m.actions[s_i]) for m, s_i in zip(automata, state, strict=True))
        for state in live
    }
    value = float(solved.delta)
    gains: list[Array] = []
    for i in range(game.n_players):
        best = jnp.asarray(-jnp.inf)
        for state in live:
            k = index[state]
            played = plays_by_state[state]
            for b in range(game.num_actions[i]):
                if b == played[i]:
                    continue
                deviated = (*played[:i], b, *played[i + 1 :])
                nxt = index[_successor(automata, state, deviated)]
                payoff = game.payoff_tensor(i)[deviated]
                best = jnp.maximum(
                    best, payoff + value * solved.values[nxt, i] - solved.values[k, i]
                )
        gains.append(jnp.asarray(best) if game.num_actions[i] > 1 else jnp.asarray(-jnp.inf))
    return jnp.stack(gains)


def is_sustainable(
    game: DenseTensorGame,
    automata: Sequence[Automaton],
    delta: float,
    *,
    tol: float | None = None,
) -> bool:
    """Is this automaton profile subgame perfect at ``delta``?"""
    threshold = base_config().tolerances.solve if tol is None else float(tol)
    return bool(jnp.max(deviation_gains(game, automata, delta)) <= threshold)


def critical_discount(
    game: DenseTensorGame,
    automata: Sequence[Automaton],
    *,
    tol: float | None = None,
) -> float:
    """The smallest δ at which the automaton profile is subgame perfect.

    Bisection on ``[0, delta_max]`` from ``config/base.yaml``. Returns ``0.0``
    when patience is not needed and ``nan`` when the profile fails even at
    ``delta_max``. Deviation gains need not be monotone in δ for exotic machines,
    so the bracket is what is reported: for grim trigger use the closed form in
    :func:`grim_critical_discount` and this as its check.
    """
    cfg = base_config().repeated
    threshold = base_config().tolerances.solve if tol is None else float(tol)

    def ok(value: float) -> bool:
        return bool(jnp.max(deviation_gains(game, automata, value)) <= threshold)

    if ok(0.0):
        return 0.0
    if not ok(cfg.delta_max):
        return float("nan")
    low, high = 0.0, float(cfg.delta_max)
    for _ in range(int(cfg.bisect_iters)):
        mid = 0.5 * (low + high)
        if ok(mid):
            high = mid
        else:
            low = mid
    return high


def _initial_sigma(
    game: DenseTensorGame,
    automata: Sequence[Automaton],
    states: Sequence[tuple[int, ...]],
    init: str | Sequence[Array],
    precision: float,
) -> tuple[Array, ...]:
    """Starting mixtures for the logit trigger solve — see ``init`` in the caller."""
    n_states = len(states)
    if isinstance(init, str):
        if init == "uniform":
            return tuple(
                jnp.full((n_states, m), 1.0 / m, dtype=jnp.float64) for m in game.num_actions
            )
        if init != "prescribed":
            raise ValueError(f"init must be 'prescribed', 'uniform' or arrays, got {init!r}")
        out = []
        for i, machine in enumerate(automata):
            prescribed = jnp.asarray(
                [int(machine.actions[state[i]]) for state in states], dtype=jnp.int32
            )
            one_hot = jnp.eye(game.num_actions[i], dtype=jnp.float64)[prescribed]
            out.append(jnp.exp(jax.nn.log_softmax(precision * one_hot, axis=-1)))
        return tuple(out)
    arrays = tuple(jnp.asarray(s, dtype=jnp.float64) for s in init)
    if len(arrays) != game.n_players:
        raise ValueError(f"init needs {game.n_players} arrays, got {len(arrays)}")
    for i, array in enumerate(arrays):
        if array.shape != (n_states, game.num_actions[i]):
            raise ValueError(
                f"init[{i}] must have shape ({n_states}, {game.num_actions[i]}), "
                f"got {tuple(array.shape)}"
            )
    return arrays


class RepeatedLogitPoint(eqx.Module):
    """A logit equilibrium of the automaton-state repeated game."""

    sigma: tuple[Array, ...]
    """Per player, ``(n_states, m_i)``: the mixed action at each machine state."""
    values: Array
    """``V_i(s)`` under those mixtures, ``(n_states, n_players)``."""
    states: Array
    initial: Array
    lam: Array
    delta: Array
    residual: Array
    n_iter: Array
    converged: Array

    @property
    def average_values(self) -> Array:
        return (1.0 - self.delta) * self.values

    @property
    def path_value(self) -> Array:
        return self.average_values[self.initial]

    def action_probability(self, player: int, action: int, state: int | None = None) -> Array:
        """Probability ``player`` plays ``action`` in ``state`` (the initial one by default)."""
        where = int(self.initial) if state is None else int(state)
        return self.sigma[int(player)][where, int(action)]


def logit_trigger_equilibrium(
    game: DenseTensorGame,
    automata: Sequence[Automaton],
    delta: float,
    lam: float,
    *,
    init: str | Sequence[Array] = "prescribed",
    damping: float | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
) -> RepeatedLogitPoint:
    """The logit-response analogue of a trigger profile.

    The automata supply the *state space and the transitions*; the actions are
    not prescribed but chosen, at every machine state, by
    ``σ_i(·|s) ∝ exp(λ Q_i(s, ·))`` with ``Q`` the continuation value of the
    deviation. λ → ∞ recovers the sharp incentive-compatibility check (the
    cooperative action goes to probability one exactly when δ exceeds the
    critical discount factor) and λ → 0 gives uniform play; in between,
    sustainability is a probability.

    The fixed point is not unique — "everyone defects" is always one of its
    solutions — so the starting point selects among them. ``init="prescribed"``
    (the default) starts at the logit-smoothed version of what the automata
    prescribe, ``softmax(λ e_a)``, which is the profile the question is about;
    ``init="uniform"`` starts from ignorance and finds whatever basin that lands
    in. An explicit sequence of arrays overrides both.

    References
    ----------
    McKelvey–Palfrey GEB 1995 applied to the automaton-state game. Tier: derived.
    """
    value = _check_delta(delta)
    precision = float(lam)
    if precision < 0:
        raise ValueError(f"lam must be >= 0, got {precision}")
    cfg = base_config()
    damp = cfg.solver.damping if damping is None else float(damping)
    threshold = cfg.tolerances.solve if tol is None else float(tol)
    limit = cfg.solver.max_iter if max_iter is None else int(max_iter)

    states, index, _ = _machine_tables(game, automata)
    n_states = len(states)
    n_players = game.n_players
    joint = _joint_profiles(game.num_actions)
    action_index = jnp.asarray(joint, dtype=jnp.int32)  # (n_joint, n_players)
    stage = jnp.stack([profile_payoffs(game, a) for a in joint])  # (n_joint, n_players)
    successors = jnp.asarray(
        [[index[_successor(automata, s, a)] for a in joint] for s in states], dtype=jnp.int32
    )  # (n_states, n_joint)
    masks = tuple(
        jnp.asarray(action_index[:, i][:, None] == jnp.arange(game.num_actions[i])[None, :])
        for i in range(n_players)
    )

    def sweep(sigma: tuple[Array, ...]) -> tuple[tuple[Array, ...], Array, Array]:
        log_sigma = tuple(jnp.log(s) for s in sigma)
        # (n_states, n_joint, n_players): log prob of each player's action in each profile.
        per_player = jnp.stack(
            [log_sigma[i][:, action_index[:, i]] for i in range(n_players)], axis=-1
        )
        total = jnp.sum(per_player, axis=-1)
        weights = jnp.exp(total)  # (n_states, n_joint)
        markov = jnp.zeros((n_states, n_states), dtype=jnp.float64)
        rows = jnp.repeat(jnp.arange(n_states), len(joint))
        markov = markov.at[rows, successors.ravel()].add(weights.ravel())
        reward = weights @ stage  # (n_states, n_players)
        values = jnp.linalg.solve(jnp.eye(n_states) - value * markov, reward)
        continuation = stage[None, :, :] + value * values[successors]  # (n_states, n_joint, n_p)
        targets = []
        for i in range(n_players):
            others = jnp.exp(total - per_player[:, :, i])  # (n_states, n_joint)
            q = (others * continuation[:, :, i]) @ masks[i]  # (n_states, m_i)
            targets.append(jnp.exp(jax.nn.log_softmax(precision * q, axis=-1)))
        return tuple(targets), values, markov

    sigma = _initial_sigma(game, automata, states, init, precision)
    residual = jnp.asarray(jnp.inf)
    values = jnp.zeros((n_states, n_players), dtype=jnp.float64)
    iterations = 0
    for step in range(1, limit + 1):
        iterations = step
        targets, values, _ = sweep(sigma)
        residual = jnp.max(
            jnp.stack([jnp.max(jnp.abs(t - s)) for t, s in zip(targets, sigma, strict=True)])
        )
        sigma = tuple((1.0 - damp) * s + damp * t for s, t in zip(sigma, targets, strict=True))
        if float(residual) < threshold:
            break
    _, values, _ = sweep(sigma)
    return RepeatedLogitPoint(
        sigma=sigma,
        values=values,
        states=jnp.asarray(states, dtype=jnp.int32),
        initial=jnp.asarray(index[tuple(m.initial for m in automata)], dtype=jnp.int32),
        lam=jnp.asarray(precision),
        delta=jnp.asarray(value),
        residual=jnp.asarray(residual),
        n_iter=jnp.asarray(iterations),
        converged=jnp.asarray(float(residual) < threshold),
    )
