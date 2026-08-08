"""The normalised (effective) game — mandatory before any decomposition.

A finite game is potential iff its *effective* payoff field is exact; full
externality symmetry on the raw tensor is sufficient but not necessary
(PROGRAMME v3 §1.1; arXiv:2405.07224 Lemma C.2). Normalisation removes each
player's own-action mean, so payoffs carry only strategically relevant
content; strategically equivalent games map to the same normalised game.

References
----------
Candogan–Menache–Ozdaglar–Parrilo, MOR 2011 (nonstrategic component);
PROGRAMME v3 §1.1 precision fix. Tier: exact (K5).
"""

from __future__ import annotations

import jax.numpy as jnp

from strataq.finite.games.tensor import DenseTensorGame


def normalise(game: DenseTensorGame) -> DenseTensorGame:
    """Subtract each player's own-action mean: û_i = u_i − mean_{a_i} u_i.

    The removed part is nonstrategic for player i (constant in own action):
    it cannot affect best responses, QRE, or the decomposition.
    """
    normalised = tuple(u - jnp.mean(u, axis=i, keepdims=True) for i, u in enumerate(game.payoffs))
    return DenseTensorGame(normalised)


def nonstrategic_part(game: DenseTensorGame) -> tuple[jnp.ndarray, ...]:
    """The per-player own-action means that normalisation removes."""
    return tuple(jnp.mean(u, axis=i, keepdims=True) for i, u in enumerate(game.payoffs))
