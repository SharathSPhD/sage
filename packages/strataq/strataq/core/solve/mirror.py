"""Magnetic mirror descent — last-iterate convergence to logit QRE.

In logit coordinates z_i (σ_i = softmax(z_i)), the magnetic proximal step with
negative-entropy mirror map and magnet at the uniform distribution is

    z_{t+1} = (z_t + η · λ ⊙ U(σ_t)) / (1 + η),

whose fixed point is z = λU(σ) — exactly the logit QRE. Entropy regularisation
makes the operator strongly monotone, giving linear *last-iterate* convergence
(no averaging), which damped iteration cannot guarantee off potentiality.

References
----------
Sokota, D'Orazio, Kolter, Loizou, Lanctot, Mitliagkas, Brown & Kroer,
"A Unified Approach to Reinforcement Learning, Quantal Response Equilibria,
and Two-Player Zero-Sum Games", ICLR 2023 (magnetic mirror descent). Tier:
exact (their result; we implement and validate).
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import Array

from strataq.core.defaults import base_config
from strataq.core.solve.fixedpoint import _as_lam_vector
from strataq.core.types import QREPoint, payoff_range
from strataq.finite.games.tensor import DenseTensorGame, expected_payoffs

_State = tuple[tuple[Array, ...], Array, Array]


@jax.jit
def _mmd_iterate(
    payoffs: tuple[Array, ...],
    z0: tuple[Array, ...],
    lam: Array,
    eta: Array,
    tol: Array,
    max_iter: Array,
) -> _State:
    game = DenseTensorGame(payoffs)

    def sigma_of(z: tuple[Array, ...]) -> tuple[Array, ...]:
        return tuple(jnp.exp(jax.nn.log_softmax(zi)) for zi in z)

    def cond(state: _State) -> Array:
        _, it, res = state
        return (res >= tol) & (it < max_iter)

    def body(state: _State) -> _State:
        z, it, _ = state
        sigma = sigma_of(z)
        utilities = expected_payoffs(game, sigma)
        z_new = tuple(
            (zi + eta * lam[i] * u) / (1.0 + eta)
            for i, (zi, u) in enumerate(zip(z, utilities, strict=True))
        )
        new_sigma = sigma_of(z_new)
        res = jnp.max(
            jnp.stack([jnp.max(jnp.abs(a - b)) for a, b in zip(new_sigma, sigma, strict=True)])
        )
        return (z_new, it + 1, res)

    return jax.lax.while_loop(cond, body, (z0, jnp.asarray(0), jnp.asarray(jnp.inf)))


def logit_qre_mirror(
    game: DenseTensorGame,
    lam: float | Array,
    *,
    eta: float | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
) -> QREPoint:
    """Solve logit QRE by magnetic mirror descent (last-iterate).

    ``eta`` defaults to the configured solver damping (both are step-size
    knobs in (0, 1]); tolerance semantics match :func:`logit_qre` — the
    residual is the sup-norm change of σ between iterates.
    """
    cfg = base_config()
    eta = cfg.solver.damping if eta is None else eta
    tol = cfg.tolerances.solve if tol is None else tol
    max_iter = cfg.solver.max_iter if max_iter is None else max_iter

    lam_vec = _as_lam_vector(lam, game.n_players)
    z0 = tuple(jnp.zeros((m,)) for m in game.num_actions)
    z, n_iter, residual = _mmd_iterate(
        game.payoffs,
        z0,
        lam_vec,
        jnp.asarray(float(eta)),
        jnp.asarray(float(tol)),
        jnp.asarray(int(max_iter)),
    )
    sigma = tuple(jnp.exp(jax.nn.log_softmax(zi)) for zi in z)
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


def solve(
    game: DenseTensorGame,
    lam: float | Array,
    *,
    method: str = "damped",
    **kwargs: float | int | Sequence[Array] | None,
) -> QREPoint:
    """Strategy dispatch: config-selectable solver by name."""
    from strataq.core.solve.fixedpoint import logit_qre  # local: avoid cycle

    solvers = {"damped": logit_qre, "mirror": logit_qre_mirror}
    if method not in solvers:
        raise ValueError(f"unknown solver '{method}'; available: {sorted(solvers)}")
    result: QREPoint = solvers[method](game, lam, **kwargs)  # type: ignore[operator]
    return result
