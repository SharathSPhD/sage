"""Engine-agnostic linear algebra helpers shared by finite and population engines."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def helmert_basis(m: int) -> Array:
    """Orthonormal basis of the mean-zero subspace of R^m, as an (m, m−1) matrix.

    Column k (k = 1..m−1) is (1,...,1, −k, 0,...,0)/√(k(k+1)) — deterministic,
    exactly orthogonal to 1, QᵀQ = I. The backbone of every tangent-space
    projection in the library (a rank-deficiency slip fakes criticality).
    """
    cols = []
    for k in range(1, m):
        v = jnp.zeros(m).at[:k].set(1.0).at[k].set(-float(k))
        cols.append(v / jnp.sqrt(float(k * (k + 1))))
    return jnp.stack(cols, axis=1)
