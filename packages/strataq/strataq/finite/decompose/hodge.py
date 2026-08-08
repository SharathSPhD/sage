"""Hodge decomposition of finite games and the canonical harmonic fraction α.

Runs on the **normalised** game, always (PROGRAMME v3 §1.1: full externality
symmetry is sufficient but not necessary on the raw game). In the subset-
component basis of :mod:`kron`, a normalised game has u_i supported on
sectors T ∋ i, and:

- **potential sector content**: u_i^T equal for all i ∈ T to a common Φ^T
  (then Φ = Σ_T Φ^T is an exact potential for the normalised game);
- **harmonic sector content**: Σ_{i∈T} m_i u_i^T = 0 — the divergence-free
  condition of Candogan's flow decomposition (the m_i weights come from the
  response-graph degrees; they matter exactly when action counts differ).

The projection of {u_i^T}_{i∈T} onto the potential subspace along the
harmonic subspace is the m-weighted mean Φ^T = Σ_{i∈T} m_i u_i^T / Σ_{i∈T} m_i.

α := ‖u^H‖ / (‖u^P‖ + ‖u^H‖) with ‖·‖ the joint Frobenius norm over players
— intrinsic, basis-independent, and computed in near-linear time.

References
----------
Candogan–Menache–Ozdaglar–Parrilo, MOR 2011 (K5, tier: exact — the
decomposition is theirs; the separable computation is the §1.2 engineering
result). α definition: PROGRAMME v3 §3.2.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from strataq.finite.decompose.kron import all_subsets, subset_component
from strataq.finite.games.normalise import normalise
from strataq.finite.games.tensor import DenseTensorGame


class HodgeDecomposition(eqx.Module):
    """potential ⊕ harmonic ⊕ nonstrategic, with norms and α."""

    potential: DenseTensorGame
    harmonic: DenseTensorGame
    nonstrategic: DenseTensorGame
    potential_norm: Array
    harmonic_norm: Array
    alpha: Array


def _joint_norm(payoffs: tuple[Array, ...]) -> Array:
    return jnp.sqrt(sum(jnp.sum(u**2) for u in payoffs))


def hodge_decompose(game: DenseTensorGame) -> HodgeDecomposition:
    """Decompose a game; the potential/harmonic split is of the normalised game."""
    n = game.n_players
    m = game.num_actions
    norm_game = normalise(game)
    nonstrategic = tuple(u - v for u, v in zip(game.payoffs, norm_game.payoffs, strict=True))

    potential = [jnp.zeros_like(u) for u in norm_game.payoffs]
    harmonic = [jnp.zeros_like(u) for u in norm_game.payoffs]

    for subset in all_subsets(n):
        if not subset:
            continue
        members = sorted(subset)
        components = {i: subset_component(norm_game.payoffs[i], subset) for i in members}
        weight_total = sum(m[i] for i in members)
        phi = sum(m[i] * components[i] for i in members) / weight_total
        for i in members:
            potential[i] = potential[i] + phi
            harmonic[i] = harmonic[i] + (components[i] - phi)

    p_norm = _joint_norm(tuple(potential))
    h_norm = _joint_norm(tuple(harmonic))
    return HodgeDecomposition(
        potential=DenseTensorGame(tuple(potential)),
        harmonic=DenseTensorGame(tuple(harmonic)),
        nonstrategic=DenseTensorGame(tuple(nonstrategic)),
        potential_norm=p_norm,
        harmonic_norm=h_norm,
        alpha=h_norm / (p_norm + h_norm),
    )


def alpha(game: DenseTensorGame) -> float:
    """The harmonic fraction of the normalised game, in [0, 1]."""
    return float(hodge_decompose(game).alpha)


def exact_potential_of(decomposition: HodgeDecomposition) -> Array:
    """Reconstruct Φ for the potential component: Σ_T Φ^T (any player's copy).

    Valid because in the potential component every player's sector content
    equals Φ^T; player 0's payoff restricted to sectors containing 0 misses
    sectors without player 0, so rebuild from all players' sector union.
    """
    payoffs = decomposition.potential.payoffs
    n = len(payoffs)
    shape = payoffs[0].shape
    phi = jnp.zeros(shape)
    for subset in all_subsets(n):
        if not subset:
            continue
        owner = min(subset)
        phi = phi + subset_component(payoffs[owner], subset)
    return phi
