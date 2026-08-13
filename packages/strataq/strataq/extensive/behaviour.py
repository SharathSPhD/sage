"""Behaviour strategies, tree passes, and the realisation equivalence to mixed.

Two passes do all the work and everything else is built from them. The *forward*
pass carries reach probabilities down the tree; the *backward* pass carries
expected payoffs up it. Both walk depth by depth, so a tree of a few hundred
nodes costs a few dozen array operations rather than a few hundred.

Kuhn's theorem is here as a tested utility rather than a remark: under perfect
recall, a behaviour strategy and its realisation-equivalent mixed strategy induce
the same distribution over terminal nodes against *every* opponent profile, and
:func:`realisation_gap` measures that on the tree you hand it.

References
----------
Kuhn 1953 (behaviour ≡ mixed under perfect recall); Osborne–Rubinstein 1994
§11.4. Tier: exact.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import Array

from strataq.core.defaults import base_config
from strataq.extensive.tree import ExtensiveGame, perfect_recall_violations
from strataq.finite.games.tensor import DenseTensorGame

__all__ = [
    "behaviour_to_mixed",
    "expected_payoffs",
    "mixed_to_behaviour",
    "node_values",
    "policy_from_behaviour",
    "pure_strategies",
    "reach_probabilities",
    "realisation_gap",
    "realisation_plan",
    "reduced_normal_form",
    "uniform_behaviour",
]


def uniform_behaviour(game: ExtensiveGame) -> Array:
    """``(n_infosets, max_actions)``: uniform over each information set's actions."""
    mask = game.action_mask()
    counts = game.infoset_actions[:, None]
    return jnp.where(mask, 1.0 / counts, 0.0)


def policy_from_behaviour(game: ExtensiveGame, behaviour: Array) -> Array:
    """``(n_nodes, max_actions)``: behaviour at decision nodes, chance at chance nodes."""
    profile = jnp.asarray(behaviour, dtype=jnp.float64)
    if profile.shape != (game.n_infosets, game.max_actions):
        raise ValueError(
            f"behaviour must have shape ({game.n_infosets}, {game.max_actions}), "
            f"got {tuple(profile.shape)}"
        )
    safe = jnp.where(game.infoset >= 0, game.infoset, 0)
    decision = profile[safe]
    policy = jnp.where(game.is_decision[:, None], decision, 0.0)
    return jnp.where(game.is_chance[:, None], game.chance, policy)


def reach_probabilities(
    game: ExtensiveGame, policy: Array, *, levels: tuple[Array, ...] | None = None
) -> Array:
    """``ρ(x)``: probability of reaching each node, ``(n_nodes,)``.

    Includes chance, so ``ρ`` at a terminal node is the probability that the play
    ends there. ``levels`` lets a caller hoist the (concrete) depth grouping out
    of a traced loop — the solvers do, because ``jnp.flatnonzero`` needs concrete
    values and a fixed-point body is traced.
    """
    order = game.levels() if levels is None else levels
    reach = jnp.zeros((game.n_nodes,), dtype=jnp.float64).at[0].set(1.0)
    for nodes in order[1:]:
        parents = game.parent[nodes]
        actions = game.entry_action[nodes]
        reach = reach.at[nodes].set(reach[parents] * policy[parents, actions])
    return reach


def node_values(
    game: ExtensiveGame, policy: Array, *, internal: tuple[Array, ...] | None = None
) -> Array:
    """``V_i(x)``: expected payoff to each player from each node, ``(n_nodes, n_players)``.

    ``internal`` is the non-terminal depth grouping, hoisted out of traced loops
    for the same reason as in :func:`reach_probabilities`.
    """
    order = game.internal_levels() if internal is None else internal
    values = jnp.zeros((game.n_nodes, game.n_players), dtype=jnp.float64)
    values = jnp.where(game.is_terminal[:, None], game.payoffs, values)
    for nodes in reversed(order):
        if int(nodes.shape[0]) == 0:
            continue
        kids = game.children[nodes]
        mask = kids >= 0
        safe = jnp.where(mask, kids, 0)
        weights = jnp.where(mask, policy[nodes], 0.0)
        values = values.at[nodes].set(jnp.sum(weights[:, :, None] * values[safe], axis=1))
    return values


def expected_payoffs(
    game: ExtensiveGame, behaviour: Array, *, internal: tuple[Array, ...] | None = None
) -> Array:
    """Expected payoff to each player under a behaviour profile, ``(n_players,)``."""
    policy = policy_from_behaviour(game, behaviour)
    return node_values(game, policy, internal=internal)[0]


def realisation_plan(game: ExtensiveGame, behaviour: Array, player: int) -> Array:
    """``(n_nodes,)``: the product of ``player``'s own action probabilities to each node.

    The sequence-form realisation weight — everyone else's moves and chance are
    left out, which is what makes it the right object for Kuhn's theorem.
    """
    policy = policy_from_behaviour(game, behaviour)
    own = jnp.where(game.player == int(player), policy, 1.0)
    plan = jnp.zeros((game.n_nodes,), dtype=jnp.float64).at[0].set(1.0)
    for nodes in game.levels()[1:]:
        parents = game.parent[nodes]
        actions = game.entry_action[nodes]
        plan = plan.at[nodes].set(plan[parents] * own[parents, actions])
    return plan


def pure_strategies(game: ExtensiveGame, player: int) -> tuple[tuple[int, ...], ...]:
    """Every pure plan for ``player``: one action per information set, in index order."""
    sets = game.player_infosets(player)
    counts = [int(game.infoset_actions[h]) for h in sets]
    size = 1
    for count in counts:
        size *= count
    limit = base_config().extensive.max_pure_strategies
    if size > limit:
        raise ValueError(
            f"player {player} has {size} pure strategies, above the limit {limit}; "
            "the reduced normal form is not the right tool for this tree."
        )
    return tuple(itertools.product(*(range(count) for count in counts)))


def _behaviour_from_pure(game: ExtensiveGame, plans: Sequence[Sequence[int]]) -> Array:
    """Turn one pure plan per player into a degenerate behaviour profile."""
    behaviour = jnp.zeros((game.n_infosets, game.max_actions), dtype=jnp.float64)
    for player, plan in enumerate(plans):
        for slot, h in enumerate(game.player_infosets(player)):
            behaviour = behaviour.at[h, int(plan[slot])].set(1.0)
    return behaviour


def reduced_normal_form(game: ExtensiveGame) -> DenseTensorGame:
    """The normal form over pure plans, with chance integrated out.

    Payoffs are computed by one vectorised tree pass over every profile at once,
    so a 64x64 plan space (Kuhn poker) is a single batched evaluation rather than
    four thousand tree walks.
    """
    plans = [pure_strategies(game, i) for i in range(game.n_players)]
    shape = tuple(len(p) for p in plans)
    profiles = list(itertools.product(*(range(n) for n in shape)))
    behaviours = jnp.stack(
        [
            _behaviour_from_pure(game, [plans[i][profile[i]] for i in range(game.n_players)])
            for profile in profiles
        ]
    )
    internal = game.internal_levels()
    payoffs = jax.vmap(lambda b: expected_payoffs(game, b, internal=internal))(behaviours)
    return DenseTensorGame(tuple(payoffs[:, i].reshape(shape) for i in range(game.n_players)))


def behaviour_to_mixed(game: ExtensiveGame, behaviour: Array, player: int) -> Array:
    """The mixed strategy realisation-equivalent to a behaviour strategy.

    The easy direction of Kuhn's theorem: the weight on a pure plan is the
    product of the behaviour probabilities it prescribes.
    """
    profile = jnp.asarray(behaviour, dtype=jnp.float64)
    plans = pure_strategies(game, player)
    sets = game.player_infosets(player)
    weights = jnp.stack(
        [
            jnp.prod(jnp.stack([profile[h, int(a)] for h, a in zip(sets, plan, strict=True)]))
            if sets
            else jnp.asarray(1.0)
            for plan in plans
        ]
    )
    return weights / jnp.sum(weights)


def _plan_allows(game: ExtensiveGame, plan: Sequence[int], player: int, infoset: int) -> bool:
    """Does this pure plan leave information set ``infoset`` reachable for ``player``?"""
    sets = game.player_infosets(player)
    choice = {h: int(a) for h, a in zip(sets, plan, strict=True)}
    node = game.infoset_members(infoset)[0]
    current = int(node)
    while int(game.parent[current]) >= 0:
        up = int(game.parent[current])
        if int(game.player[up]) == player:
            h = int(game.infoset[up])
            if choice.get(h, -1) != int(game.entry_action[current]):
                return False
        current = up
    return True


def mixed_to_behaviour(game: ExtensiveGame, mixed: Array, player: int) -> Array:
    """The behaviour strategy realisation-equivalent to a mixed strategy.

    The hard direction of Kuhn's theorem, and the one that needs perfect recall:
    ``β_h(a)`` is the mixed weight on plans that reach ``h`` and choose ``a``,
    divided by the weight on plans that reach ``h`` at all. Information sets the
    mixed strategy never reaches get the uniform distribution, which is the usual
    convention and is realisation-irrelevant.
    """
    violations = perfect_recall_violations(game)
    if violations:
        raise ValueError(
            "mixed_to_behaviour needs perfect recall; these information sets violate it: "
            f"{list(violations)}"
        )
    weights = jnp.asarray(mixed, dtype=jnp.float64).ravel()
    plans = pure_strategies(game, player)
    if weights.shape != (len(plans),):
        raise ValueError(f"mixed must have {len(plans)} entries, got {int(weights.shape[0])}")
    behaviour = uniform_behaviour(game)
    sets = game.player_infosets(player)
    for slot, h in enumerate(sets):
        reaching = jnp.asarray(
            [1.0 if _plan_allows(game, plan, player, h) else 0.0 for plan in plans]
        )
        total = float(jnp.sum(reaching * weights))
        if total <= 0.0:
            continue
        count = int(game.infoset_actions[h])
        row = jnp.zeros((game.max_actions,), dtype=jnp.float64)
        for a in range(count):
            picks = jnp.asarray([1.0 if int(plan[slot]) == a else 0.0 for plan in plans])
            row = row.at[a].set(float(jnp.sum(reaching * picks * weights)) / total)
        behaviour = behaviour.at[h].set(row)
    return behaviour


def realisation_gap(
    game: ExtensiveGame,
    behaviour: Array,
    player: int,
    *,
    n_probes: int = 8,
    seed: int | None = None,
) -> float:
    """How far the mixed image of ``behaviour`` is from realisation-equivalent.

    Draws random behaviour profiles for the other players and compares the
    expected payoffs of the behaviour strategy with those of the mixed strategy
    evaluated through the reduced normal form. Zero to machine precision is
    Kuhn's theorem holding on this tree; the test suite asserts exactly that.
    """
    root = base_config().seeds.root if seed is None else int(seed)
    key = jax.random.PRNGKey(root)
    mixed = behaviour_to_mixed(game, behaviour, player)
    normal = reduced_normal_form(game)
    worst = 0.0
    for _ in range(int(n_probes)):
        key, subkey = jax.random.split(key)
        others = _random_behaviour(game, subkey, exclude=player)
        combined = _merge(game, behaviour, others, player)
        direct = float(expected_payoffs(game, combined)[player])
        mixes = []
        for i in range(game.n_players):
            if i == player:
                mixes.append(mixed)
            else:
                mixes.append(behaviour_to_mixed(game, others, i))
        utilities = normal.payoff_tensor(player)
        for i in reversed(range(game.n_players)):
            utilities = jnp.tensordot(utilities, mixes[i], axes=([i], [0]))
        worst = max(worst, abs(direct - float(utilities)))
    return worst


def _random_behaviour(game: ExtensiveGame, key: Array, *, exclude: int) -> Array:
    mask = game.action_mask()
    draws = jax.random.uniform(key, (game.n_infosets, game.max_actions))
    draws = jnp.where(mask, draws, 0.0)
    normalised = draws / jnp.sum(draws, axis=1, keepdims=True)
    keep = (game.infoset_player == int(exclude))[:, None]
    return jnp.where(keep, uniform_behaviour(game), normalised)


def _merge(game: ExtensiveGame, mine: Array, theirs: Array, player: int) -> Array:
    keep = (game.infoset_player == int(player))[:, None]
    return jnp.where(keep, mine, theirs)
