"""Thermodynamic interpretation layer over the exact dynamics machinery.

Tier discipline (thermo/CLAUDE.md): everything in this module is `exact` —
identities of the stationary Markov chain, no analogy. The Gibbs
correspondence statements hold for potential games only; for non-potential
games what survives is the NESS reading (currents, EPR).

References
----------
K3 (Blume/Monderer–Shapley); Schnakenberg network theory; Hatano–Sasa 2001.
Tier: exact.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from strataq.core.dynamics.currents import max_current, probability_currents
from strataq.core.dynamics.entropy import entropy_production_rate
from strataq.core.dynamics.markov import glauber_generator, stationary_distribution
from strataq.finite.games.tensor import DenseTensorGame


class ThermoReading(eqx.Module):
    """One full read of the non-equilibrium meters at (game, λ)."""

    pi: Array  # stationary distribution over profiles
    currents: Array  # J* antisymmetric matrix
    max_current: Array  # sup-norm circulation
    epr: Array  # exact entropy production rate
    detailed_balance: Array  # bool: J* ≈ 0 at the identity tolerance


def thermo_read(game: DenseTensorGame, lam: float, *, db_tol: float = 1e-10) -> ThermoReading:
    """Assemble generator → π → J* → EPR in one pass (dense, small games)."""
    gen = glauber_generator(game, lam)
    pi = stationary_distribution(gen)
    currents = probability_currents(gen, pi)
    j_max = max_current(gen, pi)
    epr = entropy_production_rate(gen, pi)
    return ThermoReading(
        pi=pi,
        currents=currents,
        max_current=j_max,
        epr=epr,
        detailed_balance=j_max < db_tol,
    )


def gibbs_distribution(phi: Array, lam: float) -> Array:
    """π ∝ exp(λΦ) over profiles (flattened in lexicographic profile order)."""
    flat = lam * phi.reshape(-1)
    flat = flat - jnp.max(flat)
    weights = jnp.exp(flat)
    return weights / jnp.sum(weights)
