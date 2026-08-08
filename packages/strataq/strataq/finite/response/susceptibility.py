"""The response instruments' engine room: S, B, χ^part, χ^eq.

χ^part = λC is the opponents-frozen static FDT (K7, tier: exact).
χ^eq = (I − SB)⁻¹S is Result 1 (the strategic resolvent, tier: derived;
likely folklore — cite defensively). Everything is computed in the explicit
tangent basis; near-critical calls carry a warning flag rather than silently
returning a huge number.

References
----------
PROGRAMME v3 §3.1 (K7), §3.3 (Result 1), §8.5 (numerics rules).
Tiers: K7 exact; Result 1 derived (memory/claims.md R1).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from strataq.core.defaults import base_config
from strataq.core.types import QREPoint
from strataq.finite.games.tensor import DenseTensorGame, cross_payoff_block
from strataq.finite.response.tangent import block_basis, from_tangent, to_tangent


class ResponseOperators(eqx.Module):
    """S, B and the tangent basis assembled at a QRE point."""

    s_full: Array  # blockdiag(λ_i C_i), full coordinates
    b_full: Array  # cross-payoff operator, zero diagonal blocks
    q_basis: Array  # orthonormal tangent basis
    num_actions: tuple[int, ...]

    @property
    def s_tangent(self) -> Array:
        return to_tangent(self.q_basis, self.s_full)

    @property
    def b_tangent(self) -> Array:
        return to_tangent(self.q_basis, self.b_full)


def choice_covariance(sigma: Array) -> Array:
    """C = diag(σ) − σσᵀ — rank-deficient by construction (rows sum to zero)."""
    return jnp.diag(sigma) - jnp.outer(sigma, sigma)


def build_operators(game: DenseTensorGame, point: QREPoint) -> ResponseOperators:
    """Assemble S = blockdiag(λ_i C_i) and B_ij = ∂U_i/∂σ_j at the QRE point."""
    num_actions = game.num_actions
    total = sum(num_actions)
    offsets = [0]
    for m in num_actions:
        offsets.append(offsets[-1] + m)

    s_full = jnp.zeros((total, total))
    b_full = jnp.zeros((total, total))
    for i in range(game.n_players):
        sl_i = slice(offsets[i], offsets[i + 1])
        s_full = s_full.at[sl_i, sl_i].set(point.lam[i] * choice_covariance(point.sigma[i]))
        for j in range(game.n_players):
            if i == j:
                continue
            sl_j = slice(offsets[j], offsets[j + 1])
            b_full = b_full.at[sl_i, sl_j].set(cross_payoff_block(game, point.sigma, i, j))
    return ResponseOperators(
        s_full=s_full, b_full=b_full, q_basis=block_basis(num_actions), num_actions=num_actions
    )


def chi_partial(point: QREPoint) -> Array:
    """χ^part = blockdiag(λ_i C_i): opponents-frozen susceptibility (K7, exact)."""
    blocks = [point.lam[i] * choice_covariance(s) for i, s in enumerate(point.sigma)]
    total = sum(b.shape[0] for b in blocks)
    out = jnp.zeros((total, total))
    row = 0
    for b in blocks:
        m = b.shape[0]
        out = out.at[row : row + m, row : row + m].set(b)
        row += m
    return out


class EquilibriumResponse(eqx.Module):
    """χ^eq with its tangent representation and conditioning diagnostics."""

    chi_full: Array  # Q χ_T Qᵀ — full-coordinates equilibrium susceptibility
    chi_tangent: Array  # (I − S_T B_T)⁻¹ S_T on T
    rho_sb: Array  # spectral radius of S_T B_T
    distance_to_criticality: Array  # 1 − ρ
    near_critical: Array  # bool — warn, don't trust magnitudes


def chi_equilibrium(
    game: DenseTensorGame, point: QREPoint, *, warn_below: float | None = None
) -> EquilibriumResponse:
    """χ^eq = (I − SB)⁻¹S on the tangent space (Result 1, tier: derived).

    dσ = S(dh + B dσ) ⟹ dσ/dh = (I − SB)⁻¹S. All algebra on T via the
    Helmert basis; the full-coordinates lift is Q χ_T Qᵀ.
    """
    cfg = base_config()
    warn_below = cfg.criticality.warn_below if warn_below is None else warn_below

    ops = build_operators(game, point)
    s_t = ops.s_tangent
    b_t = ops.b_tangent
    dim = s_t.shape[0]

    eigs = jnp.linalg.eigvals(s_t @ b_t)
    rho = jnp.max(jnp.abs(eigs))
    chi_t = jnp.linalg.solve(jnp.eye(dim) - s_t @ b_t, s_t)
    return EquilibriumResponse(
        chi_full=from_tangent(ops.q_basis, chi_t),
        chi_tangent=chi_t,
        rho_sb=rho,
        distance_to_criticality=1.0 - rho,
        near_critical=(1.0 - rho) < warn_below,
    )


def chi_fd(
    game: DenseTensorGame,
    lam: float | Array,
    *,
    step: float | None = None,
    solver_tol: float | None = None,
    solver_max_iter: int | None = None,
) -> Array:
    """Finite-difference χ^eq: central differences of the solved QRE w.r.t. h.

    The oracle check for Result 1 (gate ``oracle_agreement``): perturb each
    payoff coordinate h_i(a) (own-payoff shift for player i at action a),
    re-solve, difference. O(n) solves — test-scale only.
    """
    cfg = base_config()
    step = cfg.tolerances.fd if step is None else step
    tol = cfg.tolerances.solve * 1e-3 if solver_tol is None else solver_tol
    max_iter = cfg.solver.max_iter * 20 if solver_max_iter is None else solver_max_iter

    from strataq.core.solve.fixedpoint import logit_qre  # local import: avoid cycle

    num_actions = game.num_actions
    total = sum(num_actions)
    offsets = [0]
    for m in num_actions:
        offsets.append(offsets[-1] + m)

    def perturbed(col: int, sign: float) -> Array:
        player = max(i for i in range(len(offsets) - 1) if offsets[i] <= col)
        action = col - offsets[player]
        payoffs = list(game.payoffs)
        bump_shape = [1] * game.n_players
        bump_shape[player] = num_actions[player]
        bump = jnp.zeros(num_actions[player]).at[action].set(sign * step).reshape(bump_shape)
        payoffs[player] = payoffs[player] + bump
        point = logit_qre(DenseTensorGame(tuple(payoffs)), lam, tol=tol, max_iter=max_iter)
        return jnp.concatenate(point.sigma)

    cols = []
    for col in range(total):
        cols.append((perturbed(col, +1.0) - perturbed(col, -1.0)) / (2.0 * step))
    return jnp.stack(cols, axis=1)
