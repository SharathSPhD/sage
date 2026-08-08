"""Core value types: games, QRE points, diagnostics.

Immutable equinox modules; arrays are float64 (enforced at package import).

References
----------
PROGRAMME v3 §8.2 (module map), §8.4 (QREPoint contents), §8.5 (payoff scale
folding: report both ``lam`` and ``lambda_normalised``). Engineering layer.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jax import Array


@runtime_checkable
class Game(Protocol):
    """Structural interface every engine-1 game satisfies."""

    @property
    def n_players(self) -> int: ...

    @property
    def num_actions(self) -> tuple[int, ...]: ...

    def payoff_tensor(self, player: int) -> Array:
        """Payoff to ``player`` as an array of shape ``num_actions``."""
        ...


class QREPoint(eqx.Module):
    """A solved logit quantal response equilibrium.

    ``lam`` is the user-facing precision; ``lambda_normalised`` folds the
    payoff scale in (``lam * payoff_range``) — λ is not scale-free and users
    will misread it, so both are always carried (PROGRAMME v3 §8.5).
    """

    sigma: tuple[Array, ...]
    lam: Array  # shape (n_players,)
    expected_payoffs: tuple[Array, ...]
    residual: Array  # scalar: sup-norm fixed-point violation
    n_iter: Array  # scalar int
    payoff_range: Array  # scalar: max-min over all payoff tensors
    converged: Array  # scalar bool

    @property
    def lambda_normalised(self) -> Array:
        return self.lam * self.payoff_range


class SpectrumInfo(eqx.Module):
    """Spectral read-out of SB on the tangent space.

    ``bifurcation_type`` codes: 0 = none (ρ comfortably below 1),
    1 = fold/pitchfork (real eigenvalue near/above 1),
    2 = hopf (complex pair near/above the unit circle).
    """

    eigenvalues: Array  # complex, tangent-space spectrum of SB
    rho: Array  # spectral radius
    distance_to_criticality: Array  # 1 - rho
    bifurcation_type: Array  # int code, see docstring
    near_critical: Array  # bool: distance below configured warn threshold


BIFURCATION_NONE = 0
BIFURCATION_FOLD = 1
BIFURCATION_HOPF = 2


def payoff_range(payoffs: tuple[Array, ...]) -> Array:
    """The global payoff scale: max − min across all players' tensors."""
    top = jnp.max(jnp.stack([jnp.max(u) for u in payoffs]))
    bottom = jnp.min(jnp.stack([jnp.min(u) for u in payoffs]))
    return top - bottom
