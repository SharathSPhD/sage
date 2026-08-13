"""Game trees: nodes, chance moves, information sets, payoffs at leaves.

The tree is stored as flat arrays in preorder, so a parent's index is always
below its children's and every pass over the tree is a loop over *depths*
(usually fewer than ten) rather than over nodes. That is what makes the AQRE
fixed point fast enough to be interactive on the trees people actually draw.

The authoring surface is a nested dict, which doubles as the JSON
representation: :meth:`ExtensiveGame.from_dict` and :meth:`ExtensiveGame.to_dict`
round-trip, so a tree drawn in a browser and a tree written in a test are the
same object.

A node is one of three things. A **decision** node has ``player`` and
``actions``; nodes sharing an ``infoset`` key are indistinguishable to that
player. A **chance** node has ``player: "chance"`` and ``probs``. A **terminal**
node has ``payoffs`` and nothing else.

References
----------
Kuhn 1953 (extensive form with information sets and behaviour strategies);
Selten 1975; Osborne–Rubinstein 1994 §11. Tier: exact — these are definitions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from strataq.core.defaults import base_config

__all__ = ["CHANCE", "TERMINAL", "ExtensiveGame", "perfect_recall_violations"]

CHANCE = -1
"""``player`` code for a chance node."""
TERMINAL = -2
"""``player`` code for a terminal node."""


class ExtensiveGame(eqx.Module):
    """A finite extensive-form game with chance moves and information sets.

    Nodes are numbered in preorder: node 0 is the root and every child has a
    higher index than its parent.
    """

    player: Array
    """``(n_nodes,)``: player index, or :data:`CHANCE` / :data:`TERMINAL`."""
    parent: Array
    """``(n_nodes,)``: parent index; ``-1`` at the root."""
    entry_action: Array
    """``(n_nodes,)``: the action at the parent that reaches this node; ``-1`` at the root."""
    children: Array
    """``(n_nodes, max_actions)``: child indices, ``-1`` where the action does not exist."""
    n_actions: Array
    """``(n_nodes,)``: number of actions; ``0`` at terminals."""
    infoset: Array
    """``(n_nodes,)``: information set index; ``-1`` at chance and terminal nodes."""
    chance: Array
    """``(n_nodes, max_actions)``: chance probabilities, valid only at chance nodes."""
    payoffs: Array
    """``(n_nodes, n_players)``: payoffs, valid only at terminal nodes."""
    depth: Array
    """``(n_nodes,)``: distance from the root."""
    infoset_player: Array
    """``(n_infosets,)``: who moves at each information set."""
    infoset_actions: Array
    """``(n_infosets,)``: how many actions each information set offers."""
    n_players: int = eqx.field(static=True)
    player_labels: tuple[str, ...] = eqx.field(static=True)
    infoset_labels: tuple[str, ...] = eqx.field(static=True)
    action_labels: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    """Action names per information set."""
    node_labels: tuple[str, ...] = eqx.field(static=True)
    title: str = eqx.field(static=True)

    @property
    def n_nodes(self) -> int:
        return int(self.player.shape[0])

    @property
    def n_infosets(self) -> int:
        return int(self.infoset_player.shape[0])

    @property
    def max_actions(self) -> int:
        return int(self.children.shape[1])

    @property
    def max_depth(self) -> int:
        return int(jnp.max(self.depth))

    @property
    def is_terminal(self) -> Array:
        return self.player == TERMINAL

    @property
    def is_chance(self) -> Array:
        return self.player == CHANCE

    @property
    def is_decision(self) -> Array:
        return self.player >= 0

    @property
    def has_chance(self) -> bool:
        return bool(jnp.any(self.is_chance))

    @property
    def is_perfect_information(self) -> bool:
        """True when every information set is a single node."""
        counts = jnp.zeros((self.n_infosets,), dtype=jnp.int32)
        counts = counts.at[self.infoset[self.is_decision]].add(1)
        return bool(jnp.all(counts <= 1))

    def infoset_members(self, index: int) -> tuple[int, ...]:
        """Node indices belonging to information set ``index``."""
        mask = self.is_decision & (self.infoset == int(index))
        return tuple(int(n) for n in jnp.flatnonzero(mask))

    def player_infosets(self, player: int) -> tuple[int, ...]:
        """Information sets where ``player`` moves, in index order."""
        return tuple(int(h) for h in jnp.flatnonzero(self.infoset_player == int(player)))

    def action_mask(self) -> Array:
        """``(n_infosets, max_actions)`` boolean mask of the actions that exist."""
        return jnp.arange(self.max_actions)[None, :] < self.infoset_actions[:, None]

    def levels(self) -> tuple[Array, ...]:
        """Node indices grouped by depth — the order every tree pass walks."""
        return tuple(jnp.flatnonzero(self.depth == d) for d in range(self.max_depth + 1))

    def internal_levels(self) -> tuple[Array, ...]:
        """Non-terminal node indices grouped by depth."""
        return tuple(
            jnp.flatnonzero((self.depth == d) & ~self.is_terminal)
            for d in range(self.max_depth + 1)
        )

    @staticmethod
    def from_dict(spec: dict[str, Any]) -> ExtensiveGame:
        """Build a tree from the nested dict / JSON representation.

        Parameters
        ----------
        spec
            ``{"players": [...], "root": node, "title": str}`` where a node is a
            decision (``player``, ``actions``, ``children``, optional
            ``infoset``), a chance move (``player: "chance"``, ``probs``,
            ``children``) or a leaf (``payoffs``).
        """
        return _build(spec)

    def to_dict(self) -> dict[str, Any]:
        """The nested dict this tree came from — round-trips with :meth:`from_dict`."""
        return {
            "title": self.title,
            "players": list(self.player_labels),
            "root": self._node_dict(0),
        }

    def _node_dict(self, node: int) -> dict[str, Any]:
        role = int(self.player[node])
        if role == TERMINAL:
            return {
                "label": self.node_labels[node],
                "payoffs": [float(v) for v in self.payoffs[node]],
            }
        count = int(self.n_actions[node])
        kids = [int(self.children[node, a]) for a in range(count)]
        if role == CHANCE:
            return {
                "label": self.node_labels[node],
                "player": "chance",
                "actions": [f"c{a}" for a in range(count)],
                "probs": [float(p) for p in self.chance[node, :count]],
                "children": [self._node_dict(k) for k in kids],
            }
        h = int(self.infoset[node])
        return {
            "label": self.node_labels[node],
            "player": self.player_labels[role],
            "infoset": self.infoset_labels[h],
            "actions": list(self.action_labels[h]),
            "children": [self._node_dict(k) for k in kids],
        }


class _Builder:
    """Preorder accumulator for :func:`_build`; ordinary Python, no arrays."""

    def __init__(self, players: Sequence[str]) -> None:
        self.players = tuple(str(p) for p in players)
        self.player: list[int] = []
        self.parent: list[int] = []
        self.entry: list[int] = []
        self.children: list[list[int]] = []
        self.infoset: list[int] = []
        self.chance: list[list[float]] = []
        self.payoffs: list[list[float]] = []
        self.depth: list[int] = []
        self.labels: list[str] = []
        self.infoset_keys: dict[tuple[int, str], int] = {}
        self.infoset_player: list[int] = []
        self.infoset_labels: list[str] = []
        self.action_labels: list[tuple[str, ...]] = []

    def new_node(self, parent: int, entry: int, depth: int, label: str) -> int:
        node = len(self.player)
        self.player.append(TERMINAL)
        self.parent.append(parent)
        self.entry.append(entry)
        self.children.append([])
        self.infoset.append(-1)
        self.chance.append([])
        self.payoffs.append([])
        self.depth.append(depth)
        self.labels.append(label)
        return node

    def resolve_player(self, name: Any) -> int:
        if isinstance(name, int):
            if not 0 <= name < len(self.players):
                raise ValueError(f"player index {name} out of range")
            return name
        text = str(name)
        if text not in self.players:
            raise ValueError(f"unknown player {text!r}; declared players are {list(self.players)}")
        return self.players.index(text)


def _build(spec: dict[str, Any]) -> ExtensiveGame:
    if "players" not in spec or "root" not in spec:
        raise ValueError("a tree spec needs 'players' and 'root'")
    players = [str(p) for p in spec["players"]]
    if len(players) < 1:
        raise ValueError("a tree needs at least one player")
    if len(set(players)) != len(players):
        raise ValueError(f"player names must be unique, got {players}")
    builder = _Builder(players)
    limit = base_config().extensive.max_nodes
    _walk(builder, spec["root"], parent=-1, entry=-1, depth=0, limit=limit)

    n_nodes = len(builder.player)
    max_actions = max((len(kids) for kids in builder.children), default=1) or 1
    children = jnp.full((n_nodes, max_actions), -1, dtype=jnp.int32)
    chance = jnp.zeros((n_nodes, max_actions), dtype=jnp.float64)
    payoffs = jnp.zeros((n_nodes, len(players)), dtype=jnp.float64)
    counts = []
    for node in range(n_nodes):
        kids = builder.children[node]
        counts.append(len(kids))
        if kids:
            children = children.at[node, : len(kids)].set(jnp.asarray(kids, dtype=jnp.int32))
        if builder.chance[node]:
            chance = chance.at[node, : len(kids)].set(
                jnp.asarray(builder.chance[node], dtype=jnp.float64)
            )
        if builder.payoffs[node]:
            payoffs = payoffs.at[node].set(jnp.asarray(builder.payoffs[node], dtype=jnp.float64))

    game = ExtensiveGame(
        player=jnp.asarray(builder.player, dtype=jnp.int32),
        parent=jnp.asarray(builder.parent, dtype=jnp.int32),
        entry_action=jnp.asarray(builder.entry, dtype=jnp.int32),
        children=children,
        n_actions=jnp.asarray(counts, dtype=jnp.int32),
        infoset=jnp.asarray(builder.infoset, dtype=jnp.int32),
        chance=chance,
        payoffs=payoffs,
        depth=jnp.asarray(builder.depth, dtype=jnp.int32),
        infoset_player=jnp.asarray(builder.infoset_player, dtype=jnp.int32),
        infoset_actions=jnp.asarray(
            [len(labels) for labels in builder.action_labels], dtype=jnp.int32
        ),
        n_players=len(players),
        player_labels=tuple(players),
        infoset_labels=tuple(builder.infoset_labels),
        action_labels=tuple(builder.action_labels),
        node_labels=tuple(builder.labels),
        title=str(spec.get("title", "extensive game")),
    )
    _validate(game)
    return game


def _walk(
    builder: _Builder, spec: dict[str, Any], *, parent: int, entry: int, depth: int, limit: int
) -> int:
    if len(builder.player) >= limit:
        raise ValueError(f"tree exceeds the node limit {limit} from config/base.yaml")
    label = str(spec.get("label", f"n{len(builder.player)}"))
    node = builder.new_node(parent, entry, depth, label)
    if parent >= 0:
        builder.children[parent].append(node)

    if "payoffs" in spec:
        values = [float(v) for v in spec["payoffs"]]
        if len(values) != len(builder.players):
            raise ValueError(
                f"node {label!r}: {len(values)} payoffs for {len(builder.players)} players"
            )
        builder.payoffs[node] = values
        return node

    if "children" not in spec:
        raise ValueError(f"node {label!r} has neither 'payoffs' nor 'children'")
    kids = list(spec["children"])
    if len(kids) < 1:
        raise ValueError(f"node {label!r} has an empty 'children'")
    role = spec.get("player")
    if role is None:
        raise ValueError(f"node {label!r} needs 'player' (a name, an index, or 'chance')")

    if isinstance(role, str) and role.lower() == "chance":
        probs = [float(p) for p in spec.get("probs", [1.0 / len(kids)] * len(kids))]
        if len(probs) != len(kids):
            raise ValueError(f"node {label!r}: {len(probs)} probs for {len(kids)} children")
        if any(p < 0 for p in probs):
            raise ValueError(f"node {label!r}: chance probabilities must be non-negative")
        total = sum(probs)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"node {label!r}: chance probabilities sum to {total}, not 1")
        builder.player[node] = CHANCE
        builder.chance[node] = probs
    else:
        index = builder.resolve_player(role)
        actions = [str(a) for a in spec.get("actions", [f"a{i}" for i in range(len(kids))])]
        if len(actions) != len(kids):
            raise ValueError(f"node {label!r}: {len(actions)} actions for {len(kids)} children")
        key = str(spec.get("infoset", f"__{label}__{node}"))
        handle = (index, key)
        if handle in builder.infoset_keys:
            h = builder.infoset_keys[handle]
            if builder.action_labels[h] != tuple(actions):
                raise ValueError(
                    f"information set {key!r} has actions {list(builder.action_labels[h])} "
                    f"elsewhere but {actions} at node {label!r}"
                )
        else:
            h = len(builder.infoset_player)
            builder.infoset_keys[handle] = h
            builder.infoset_player.append(index)
            builder.infoset_labels.append(key)
            builder.action_labels.append(tuple(actions))
        builder.player[node] = index
        builder.infoset[node] = h

    for slot, child in enumerate(kids):
        _walk(builder, child, parent=node, entry=slot, depth=depth + 1, limit=limit)
    return node


def _validate(game: ExtensiveGame) -> None:
    if not bool(jnp.all(jnp.isfinite(game.payoffs))):
        raise ValueError("payoffs must be finite")
    for h in range(game.n_infosets):
        members = game.infoset_members(h)
        if not members:
            raise ValueError(f"information set {game.infoset_labels[h]!r} has no nodes")
        expected = int(game.infoset_actions[h])
        for node in members:
            if int(game.n_actions[node]) != expected:
                raise ValueError(
                    f"information set {game.infoset_labels[h]!r} mixes "
                    f"{expected} and {int(game.n_actions[node])} actions"
                )


def perfect_recall_violations(game: ExtensiveGame) -> tuple[str, ...]:
    """Information sets whose nodes disagree about the player's own past.

    Perfect recall is what makes behaviour and mixed strategies interchangeable
    (Kuhn 1953), so :func:`strataq.extensive.behaviour.mixed_to_behaviour`
    refuses without it. An empty tuple means the tree has perfect recall.
    """
    bad: list[str] = []
    for h in range(game.n_infosets):
        owner = int(game.infoset_player[h])
        histories = {_own_history(game, node, owner) for node in game.infoset_members(h)}
        if len(histories) > 1:
            bad.append(game.infoset_labels[h])
    return tuple(bad)


def _own_history(game: ExtensiveGame, node: int, player: int) -> tuple[tuple[int, int], ...]:
    """The player's own (information set, action) pairs on the path to ``node``."""
    path: list[tuple[int, int]] = []
    current = node
    while int(game.parent[current]) >= 0:
        up = int(game.parent[current])
        if int(game.player[up]) == player:
            path.append((int(game.infoset[up]), int(game.entry_action[current])))
        current = up
    return tuple(reversed(path))
