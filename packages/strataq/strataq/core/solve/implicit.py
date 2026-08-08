"""Implicit differentiation through the QRE fixed point — sharing the resolvent.

The fixed point σ*(h) satisfies σ = softmax(λ(U(σ) + h)). Its derivative is
Result 1's resolvent: dσ*/dh = (I − SB)⁻¹S on the tangent space. The custom
VJP therefore solves one *transposed* tangent system per backward pass —
the same operator the susceptibility, spectrum and bifurcation detector use.
Implemented once, reused (PROGRAMME v3 §8.5).

References
----------
Implicit function theorem on the logit fixed point; Result 1 (claims R1,
tier: derived, FD-verified). Deep-equilibrium-style custom VJP.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from strataq.core.solve.fixedpoint import logit_qre
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.susceptibility import build_operators
from strataq.finite.response.tangent import block_basis


def _split(flat: Array, num_actions: tuple[int, ...]) -> tuple[Array, ...]:
    out, off = [], 0
    for m in num_actions:
        out.append(flat[off : off + m])
        off += m
    return tuple(out)


@partial(jax.custom_vjp, nondiff_argnums=(0, 2, 3, 4))
def qre_sigma(
    game: DenseTensorGame,
    h: Array,
    lam: float,
    tol: float,
    max_iter: int,
) -> Array:
    """σ*(h) as one concatenated vector, differentiable w.r.t. the payoff field h.

    ``h`` is a flat vector of per-player own-action payoff shifts (length
    Σm_i), the conjugate-field layout the instruments use.
    """
    return _forward(game, h, lam, tol, max_iter)[0]


def _forward(
    game: DenseTensorGame, h: Array, lam: float, tol: float, max_iter: int
) -> tuple[Array, Array]:
    shifts = _split(h, game.num_actions)
    payoffs = []
    for i, u in enumerate(game.payoffs):
        bump_shape = [1] * game.n_players
        bump_shape[i] = game.num_actions[i]
        payoffs.append(u + shifts[i].reshape(bump_shape))
    point = logit_qre(DenseTensorGame(tuple(payoffs)), lam, tol=tol, max_iter=max_iter)
    return jnp.concatenate(point.sigma), jnp.concatenate(point.sigma)


def _fwd(
    game: DenseTensorGame, h: Array, lam: float, tol: float, max_iter: int
) -> tuple[Array, Array]:
    sigma_flat, residual_data = _forward(game, h, lam, tol, max_iter)
    return sigma_flat, residual_data


def _bwd(
    game: DenseTensorGame,
    lam: float,
    tol: float,
    max_iter: int,
    residual_data: Array,
    cotangent: Array,
) -> tuple[Array]:
    sigma = _split(residual_data, game.num_actions)
    # Rebuild the point cheaply: S and B depend only on (sigma, lam).
    from strataq.core.types import QREPoint, payoff_range
    from strataq.finite.games.tensor import expected_payoffs

    lam_vec = jnp.full((game.n_players,), jnp.asarray(lam, dtype=jnp.float64))
    point = QREPoint(
        sigma=sigma,
        lam=lam_vec,
        expected_payoffs=expected_payoffs(game, sigma),
        residual=jnp.asarray(0.0),
        n_iter=jnp.asarray(0),
        payoff_range=payoff_range(game.payoffs),
        converged=jnp.asarray(True),
    )
    ops = build_operators(game, point)
    q = block_basis(game.num_actions)
    s_t = q.T @ ops.s_full @ q
    b_t = q.T @ ops.b_full @ q
    dim = s_t.shape[0]
    # vjp: v ↦ Sᵀ(I − SB)⁻ᵀ v on T, lifted back to full coordinates.
    v_t = q.T @ cotangent
    w_t = jnp.linalg.solve((jnp.eye(dim) - s_t @ b_t).T, v_t)
    grad_h = q @ (s_t.T @ w_t)
    return (grad_h,)


qre_sigma.defvjp(_fwd, _bwd)
