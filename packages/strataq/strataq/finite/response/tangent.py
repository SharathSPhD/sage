"""Tangent-space machinery — where ALL (I − SB) algebra must live.

C_i is rank-deficient by construction (rows and columns sum to zero). Doing
linear algebra on the full simplex coordinates silently produces a spurious
zero eigenvalue and a *false criticality reading* — the project's
highest-ranked numerical risk. Every operator here is expressed in an explicit
orthonormal basis of the mean-zero subspace T = ⊕_i {v : 1ᵀv = 0}.

References
----------
PROGRAMME v3 §3.3 (definition of T), §8.5 (the projection rule). Tier:
engineering invariant backing exact results.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from strataq.core.linalg import helmert_basis

__all__ = ["block_basis", "from_tangent", "helmert_basis", "to_tangent"]


def block_basis(num_actions: tuple[int, ...]) -> Array:
    """Block-diagonal orthonormal tangent basis Q for the product of simplices.

    Shape (Σm_i, Σ(m_i − 1)); QᵀQ = I; columns span T.
    """
    total = sum(num_actions)
    reduced = sum(m - 1 for m in num_actions)
    q_full = jnp.zeros((total, reduced))
    row = col = 0
    for m in num_actions:
        q_full = q_full.at[row : row + m, col : col + m - 1].set(helmert_basis(m))
        row += m
        col += m - 1
    return q_full


def to_tangent(q_basis: Array, matrix: Array) -> Array:
    """Represent a full-coordinates operator on T: Qᵀ M Q."""
    return q_basis.T @ matrix @ q_basis


def from_tangent(q_basis: Array, matrix_t: Array) -> Array:
    """Lift a T-representation back to full coordinates: Q M_T Qᵀ."""
    return q_basis @ matrix_t @ q_basis.T
