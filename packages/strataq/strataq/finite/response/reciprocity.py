"""The reciprocity meter — the programme's first genuinely new instrument.

ℛ = ‖χ^eq − (χ^eq)ᵀ‖_F / ‖χ^eq + (χ^eq)ᵀ‖_F.

Result 2 (N1, tier: derived): χ^eq is symmetric ⟺ S(B − Bᵀ)S = 0 ⟺
(full support) B = Bᵀ on T ⟺ the normalised game has zero harmonic
component. Strategic feedback neither creates nor destroys reciprocity, so
the *observable* equilibrium response inherits the symmetry of the
*unobservable* payoff operator exactly — which is what makes ℛ measurable
from cross-agent pass-through asymmetry, λ-free (N2).

References
----------
PROGRAMME v3 §3.3 Result 2; memory/claims.md N1/N2 (derived; prior-art sweep
2026-08-08 found no prior statement). Onsager reciprocity is the physics
antecedent (cited, not claimed).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from strataq.core.types import QREPoint
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.susceptibility import EquilibriumResponse, chi_equilibrium


def reciprocity_defect_of(chi: Array) -> Array:
    """ℛ from a response matrix (full or tangent representation — norms agree).

    ℛ ∈ [0, ∞): a norm ratio, not a fraction — values above 1 mean the
    antisymmetric (circulating) response dominates (findings F-0001).
    """
    asym = jnp.linalg.norm(chi - chi.T)
    sym = jnp.linalg.norm(chi + chi.T)
    return jnp.asarray(asym / sym)


def reciprocity_defect(
    game: DenseTensorGame, point: QREPoint, *, response: EquilibriumResponse | None = None
) -> Array:
    """ℛ at a QRE point. Reads 0 (to tolerance) iff the normalised game is potential."""
    resp = chi_equilibrium(game, point) if response is None else response
    return reciprocity_defect_of(resp.chi_tangent)
