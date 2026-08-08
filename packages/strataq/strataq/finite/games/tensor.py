"""Dense tensor games: the concrete Engine-1 game object.

Payoffs are one array per player, each of shape ``(m_1, ..., m_N)`` — the full
joint-action tensor. Dense is the exact path for small games; the matrix-free
path (never materialising B) arrives with large-grid domains.

References
----------
PROGRAMME v3 §3 notation. Engineering layer.
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from strataq.core.types import payoff_range


class DenseTensorGame(eqx.Module):
    """An N-player finite game as a tuple of payoff tensors."""

    payoffs: tuple[Array, ...]

    def __init__(self, payoffs: Sequence[Array]) -> None:
        arrays = tuple(jnp.asarray(u, dtype=jnp.float64) for u in payoffs)
        if not arrays:
            raise ValueError("a game needs at least one player's payoff tensor")
        shape = arrays[0].shape
        if len(shape) != len(arrays):
            raise ValueError(
                f"{len(arrays)} players but payoff tensors have rank {len(shape)}; "
                "each tensor must have one axis per player."
            )
        for i, u in enumerate(arrays):
            if u.shape != shape:
                raise ValueError(f"player {i} tensor shape {u.shape} != {shape}")
        self.payoffs = arrays

    @property
    def n_players(self) -> int:
        return len(self.payoffs)

    @property
    def num_actions(self) -> tuple[int, ...]:
        return tuple(self.payoffs[0].shape)

    def payoff_tensor(self, player: int) -> Array:
        return self.payoffs[player]

    @property
    def payoff_range(self) -> Array:
        return payoff_range(self.payoffs)


def expected_payoffs(game: DenseTensorGame, sigma: Sequence[Array]) -> tuple[Array, ...]:
    """U_i(a; σ_{-i}) for every player: contract each tensor with all rivals' mixes."""
    out: list[Array] = []
    for i in range(game.n_players):
        u = game.payoff_tensor(i)
        # Contract rival axes one at a time, keeping axis i in place.
        for j in range(game.n_players - 1, -1, -1):
            if j == i:
                continue
            u = jnp.tensordot(u, sigma[j], axes=([j], [0]))
        out.append(u)
    return tuple(out)


def cross_payoff_block(game: DenseTensorGame, sigma: Sequence[Array], i: int, j: int) -> Array:
    """B_ij = ∂U_i/∂σ_j as an (m_i, m_j) matrix (zero block when i == j).

    Contract u_i with σ_k for every k ∉ {i, j}, leaving axes (a_i, a_j).
    """
    if i == j:
        m = game.num_actions[i]
        return jnp.zeros((m, m))
    u = game.payoff_tensor(i)
    for k in range(game.n_players - 1, -1, -1):
        if k in (i, j):
            continue
        u = jnp.tensordot(u, sigma[k], axes=([k], [0]))
    # Remaining axes are (i, j) in original order; transpose if needed.
    return u if i < j else jnp.swapaxes(u, 0, 1)
