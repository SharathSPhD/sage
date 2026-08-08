"""The phase locator: spectrum of SB on the tangent space.

(I − SB) is simultaneously the susceptibility resolvent, the contraction
certificate (ρ(SB) < 1 ⟹ unique QRE), and the bifurcation detector (turning
points of the QRE correspondence are eigenvalues of SB crossing 1). A real
eigenvalue crossing reads fold/pitchfork (multiplicity, Brock–Durlauf); a
complex pair crossing reads Hopf (cycles). Result 3 (N3, tier: derived,
numerical verification outstanding): B = Bᵀ ⟹ real spectrum ⟹ no Hopf in
full potential games.

References
----------
PROGRAMME v3 §3.4; memory/claims.md N3 (ADR-0007).
"""

from __future__ import annotations

import jax.numpy as jnp

from strataq.core.defaults import base_config
from strataq.core.types import (
    BIFURCATION_FOLD,
    BIFURCATION_HOPF,
    BIFURCATION_NONE,
    QREPoint,
    SpectrumInfo,
)
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.susceptibility import build_operators


def strategic_spectrum(
    game: DenseTensorGame,
    point: QREPoint,
    *,
    warn_below: float | None = None,
    imag_tol: float | None = None,
) -> SpectrumInfo:
    """Eigenvalues of S_T B_T, ρ, distance to criticality, bifurcation typing.

    Typing looks at the eigenvalue(s) achieving ρ: effectively real ⟹
    fold/pitchfork route; complex pair ⟹ Hopf route; ρ comfortably below 1 ⟹
    none. ``imag_tol`` (relative) separates "effectively real" from complex —
    defaults to the oracle rung of the tolerance ladder.
    """
    cfg = base_config()
    warn_below = cfg.criticality.warn_below if warn_below is None else warn_below
    imag_tol = cfg.tolerances.oracle if imag_tol is None else imag_tol

    ops = build_operators(game, point)
    eigs = jnp.linalg.eigvals(ops.s_tangent @ ops.b_tangent)
    magnitudes = jnp.abs(eigs)
    rho = jnp.max(magnitudes)
    distance = 1.0 - rho

    leading = eigs[jnp.argmax(magnitudes)]
    leading_is_real = jnp.abs(jnp.imag(leading)) <= imag_tol * jnp.maximum(jnp.abs(leading), 1.0)
    near = distance < warn_below
    bif_type = jnp.where(
        ~near, BIFURCATION_NONE, jnp.where(leading_is_real, BIFURCATION_FOLD, BIFURCATION_HOPF)
    )
    return SpectrumInfo(
        eigenvalues=eigs,
        rho=rho,
        distance_to_criticality=distance,
        bifurcation_type=bif_type,
        near_critical=near,
    )


def critical_lambda(
    game: DenseTensorGame,
    *,
    bracket: tuple[float, float] | None = None,
    tol: float | None = None,
    max_bisect: int | None = None,
) -> float:
    """Smallest λ (uniform across players) where ρ(S_T B_T) reaches 1, by bisection.

    Along the principal branch, ρ grows from 0 at λ=0 (S ∝ λ). Returns the
    upper bracket if no crossing is found there (then the game is subcritical
    throughout the bracket).
    """
    from strataq.core.solve.fixedpoint import logit_qre  # local import: avoid cycle

    cfg = base_config()
    lo, hi = bracket if bracket is not None else (0.0, 1e3)
    tol = cfg.tolerances.oracle if tol is None else tol
    max_bisect = cfg.solver.max_iter if max_bisect is None else max_bisect

    def rho_at(lam: float) -> float:
        point = logit_qre(game, lam)
        ops = build_operators(game, point)
        return float(jnp.max(jnp.abs(jnp.linalg.eigvals(ops.s_tangent @ ops.b_tangent))))

    if rho_at(hi) < 1.0:
        return float(hi)
    for _ in range(max_bisect):
        mid = 0.5 * (lo + hi)
        if hi - lo < tol:
            break
        if rho_at(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))
