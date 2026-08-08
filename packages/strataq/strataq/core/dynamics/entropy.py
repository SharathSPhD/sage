"""Exact entropy production rate of the stationary Glauber chain.

σ_EP = ½ Σ_{a,a'} [π(a)w(a→a') − π(a')w(a'→a)] · log[π(a)w(a→a') / (π(a')w(a'→a))] ≥ 0,

zero iff detailed balance. This is the exact (generator-level) dissipation
meter; trajectory estimators (KLD, TUR, NEEP) validate against it before
touching data.

References
----------
Standard stochastic thermodynamics (Schnakenberg network theory);
PROGRAMME v3 §3.5. Tier: exact.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def entropy_production_rate(generator: Array, pi: Array) -> Array:
    """Exact EPR from the generator and its stationary distribution.

    Glauber-logit rates are strictly positive (softmax), so the chain has no
    one-way edges and σ_EP is a finite sum of nonnegative terms
    (x − y)·log(x/y) ≥ 0 — nonnegativity is structural, not clamped.
    """
    rates = generator - jnp.diag(jnp.diag(generator))
    forward = pi[:, None] * rates
    backward = forward.T
    both = (forward > 0) & (backward > 0)
    ratio = jnp.where(both, forward / jnp.where(both, backward, 1.0), 1.0)
    return 0.5 * jnp.sum(jnp.where(both, (forward - backward) * jnp.log(ratio), 0.0))
