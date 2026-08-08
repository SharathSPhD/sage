"""The separable Kronecker transform: tensor components by "which axes vary".

The game graph is K_{m_1} □ ... □ K_{m_N}; its Laplacian is a Kronecker sum
whose eigenspaces are tensor products of {constants} and {mean-zero} per
axis, indexed by the subset T of "non-constant" coordinates. Projections onto
these sectors are separable per-axis centering/averaging — an FFT-like
transform, cost O(tensor size × N), no iterative solver (PROGRAMME v3 §1.2:
the single most valuable engineering finding of the sanity checks).

References
----------
PROGRAMME v3 §1.2; Cartesian-product Laplacian spectra are textbook algebraic
graph theory. Tier: exact.
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp
from jax import Array


def center_axis(tensor: Array, axis: int) -> Array:
    """Project onto the mean-zero subspace along one axis."""
    return tensor - jnp.mean(tensor, axis=axis, keepdims=True)


def average_axis(tensor: Array, axis: int) -> Array:
    """Project onto constants along one axis (mean, broadcast back)."""
    return jnp.broadcast_to(jnp.mean(tensor, axis=axis, keepdims=True), tensor.shape)


def subset_component(tensor: Array, subset: frozenset[int]) -> Array:
    """The component of ``tensor`` varying exactly on the axes in ``subset``.

    Apply centering on axes in the subset and averaging on the rest; the per-
    axis projectors commute, so order is irrelevant and cost is one pass.
    """
    out = tensor
    for axis in range(tensor.ndim):
        out = center_axis(out, axis) if axis in subset else average_axis(out, axis)
    return out


def all_subsets(n_axes: int) -> list[frozenset[int]]:
    """Every subset of axes, ∅ first, in size order (deterministic)."""
    subsets: list[frozenset[int]] = []
    for size in range(n_axes + 1):
        for combo in itertools.combinations(range(n_axes), size):
            subsets.append(frozenset(combo))
    return subsets


def subset_decompose(tensor: Array) -> dict[frozenset[int], Array]:
    """Full orthogonal decomposition: tensor = Σ_T component_T.

    Orthogonality and exact reconstruction are tested properties, not
    assumptions.
    """
    return {t: subset_component(tensor, t) for t in all_subsets(tensor.ndim)}
