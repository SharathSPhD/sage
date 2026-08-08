"""Damped fixed-point solver for logit QRE.

σ_i ← (1−δ)σ_i + δ softmax(λ_i U_i(σ_{-i})), iterated to the configured
residual. All softmax through ``jax.nn.log_softmax`` — raw payoffs are never
exponentiated (PROGRAMME v3 §8.5).

References
----------
McKelvey–Palfrey GEB 1995 (logit QRE, K1 tier: exact); damping is standard
Krasnoselskii–Mann averaging. Contraction for small λ ⟹ unique QRE.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import Array

from strataq.core.defaults import base_config
from strataq.core.types import QREPoint, payoff_range
from strataq.finite.games.tensor import DenseTensorGame, expected_payoffs

_State = tuple[tuple[Array, ...], Array, Array]


def _as_lam_vector(lam: float | Array, n_players: int) -> Array:
    arr = jnp.asarray(lam, dtype=jnp.float64)
    if arr.ndim == 0:
        arr = jnp.full((n_players,), arr)
    if arr.shape != (n_players,):
        raise ValueError(f"lam must be scalar or shape ({n_players},), got {arr.shape}")
    return arr


def logit_response(game: DenseTensorGame, sigma: Sequence[Array], lam: Array) -> tuple[Array, ...]:
    """One synchronous logit best-response sweep."""
    utilities = expected_payoffs(game, sigma)
    return tuple(jnp.exp(jax.nn.log_softmax(lam[i] * u)) for i, u in enumerate(utilities))


@jax.jit
def _iterate(
    payoffs: tuple[Array, ...],
    sigma0: tuple[Array, ...],
    lam: Array,
    damping: Array,
    tol: Array,
    max_iter: Array,
) -> tuple[tuple[Array, ...], Array, Array]:
    """Damped fixed-point loop, JIT-compiled per action-space shape."""
    game = DenseTensorGame(payoffs)

    def cond(state: _State) -> Array:
        _, it, res = state
        return (res >= tol) & (it < max_iter)

    def body(state: _State) -> _State:
        sigma, it, _ = state
        target = logit_response(game, sigma, lam)
        res = jnp.max(
            jnp.stack([jnp.max(jnp.abs(t - s)) for t, s in zip(target, sigma, strict=True)])
        )
        new_sigma = tuple(
            (1.0 - damping) * s + damping * t for s, t in zip(sigma, target, strict=True)
        )
        return (new_sigma, it + 1, res)

    result: _State = jax.lax.while_loop(cond, body, (sigma0, jnp.asarray(0), jnp.asarray(jnp.inf)))
    return result


def logit_qre(
    game: DenseTensorGame,
    lam: float | Array,
    *,
    init: Sequence[Array] | None = None,
    damping: float | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
) -> QREPoint:
    """Solve the logit QRE fixed point by damped iteration.

    Defaults (damping, tol, max_iter) resolve from config/base.yaml — no
    inline constants. Returns a :class:`QREPoint` carrying both ``lam`` and
    ``lambda_normalised`` (payoff scale folded in).
    """
    cfg = base_config()
    damping = cfg.solver.damping if damping is None else damping
    tol = cfg.tolerances.solve if tol is None else tol
    max_iter = cfg.solver.max_iter if max_iter is None else max_iter

    n = game.n_players
    lam_vec = _as_lam_vector(lam, n)
    sigma = (
        tuple(jnp.asarray(s, dtype=jnp.float64) for s in init)
        if init is not None
        else tuple(jnp.full((m,), 1.0 / m) for m in game.num_actions)
    )

    sigma, n_iter, residual = _iterate(
        game.payoffs,
        sigma,
        lam_vec,
        jnp.asarray(float(damping)),
        jnp.asarray(float(tol)),
        jnp.asarray(int(max_iter)),
    )

    utilities = expected_payoffs(game, sigma)
    return QREPoint(
        sigma=sigma,
        lam=lam_vec,
        expected_payoffs=utilities,
        residual=jnp.asarray(residual),
        n_iter=jnp.asarray(n_iter),
        payoff_range=payoff_range(game.payoffs),
        converged=jnp.asarray(residual < tol),
    )
