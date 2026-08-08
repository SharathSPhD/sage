"""Stationary probability currents J* on the profile space.

J*(a, a') = π(a)w(a→a') − π(a')w(a'→a). Zero everywhere ⟺ detailed balance
⟺ (for Glauber-logit) exact potential game (K3). Nonzero J* is the NESS
signature: circulation on the profile lattice, the visual heart of the
project (Learn explainer 6).

References
----------
PROGRAMME v3 §3.5 (exact formulae). Tier: exact.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def probability_currents(generator: Array, pi: Array) -> Array:
    """Antisymmetric current matrix J[a, a'] on off-diagonal transitions."""
    flow = pi[:, None] * generator
    flow = flow - jnp.diag(jnp.diag(flow))
    return flow - flow.T


def max_current(generator: Array, pi: Array) -> Array:
    """Sup-norm of J*: a single-number circulation reading."""
    return jnp.max(jnp.abs(probability_currents(generator, pi)))
