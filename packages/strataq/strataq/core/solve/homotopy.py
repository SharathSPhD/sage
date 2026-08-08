"""Pseudo-arclength continuation of the logit QRE correspondence.

Traces the principal branch from the centroid at λ = 0, following
H(z_T, λ) = z_T − Qᵀ(λ·U(σ(z_T))) = 0 in tangent-logit coordinates with an
arclength constraint, Euler predictor + Newton corrector. Folds (turning
points in λ) are detected as arclength λ-direction reversals; separately,
ρ(SB) is recorded at every accepted point — the same operator as the
susceptibility resolvent — so bifurcation *proximity* is read off the branch
via ``rhos`` (there is no discrete bifurcation flag beyond ``turning_points``).

References
----------
Turocy, GEB 2005 (the method Gambit uses — our oracle); PROGRAMME v3 §3.4
(spectral typing). Tier: engineering on top of exact structure.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from strataq.core.defaults import base_config
from strataq.core.types import QREPoint, payoff_range
from strataq.finite.games.tensor import DenseTensorGame, expected_payoffs
from strataq.finite.response.susceptibility import build_operators
from strataq.finite.response.tangent import block_basis


class Branch(eqx.Module):
    """A traced piece of the QRE correspondence."""

    lambdas: Array  # (n_points,)
    sigmas: Array  # (n_points, total_actions)
    rhos: Array  # (n_points,) spectral radius of SB along the branch
    turning_points: Array  # (n_points,) bool — arclength direction reversed in λ


def _sigma_from_z(z_t: Array, q_basis: Array, num_actions: tuple[int, ...]) -> tuple[Array, ...]:
    z_full = q_basis @ z_t
    sigmas = []
    off = 0
    for m in num_actions:
        sigmas.append(jnp.exp(jax.nn.log_softmax(z_full[off : off + m])))
        off += m
    return tuple(sigmas)


def logit_branch(
    game: DenseTensorGame,
    lam_max: float,
    *,
    n_points: int | None = None,
    first_step: float | None = None,
    newton_tol: float | None = None,
) -> Branch:
    """Trace the principal branch from λ = 0 to λ_max (or until n_points).

    Pseudo-arclength: the step continues through folds (λ locally decreasing)
    instead of stalling on them; each accepted point is Newton-corrected on
    the augmented system to ``newton_tol``.
    """
    cfg = base_config()
    n_points = 200 if n_points is None else n_points
    first_step = 0.03 if first_step is None else first_step
    newton_tol = cfg.tolerances.solve if newton_tol is None else newton_tol

    num_actions = game.num_actions
    q_basis = block_basis(num_actions)
    dim = q_basis.shape[1]

    def residual(z_t: Array, lam: float | Array) -> Array:
        sigma = _sigma_from_z(z_t, q_basis, num_actions)
        utilities = expected_payoffs(game, sigma)
        target = jnp.concatenate([lam * u for u in utilities])
        return z_t - q_basis.T @ target

    res_jac_z = jax.jacfwd(residual, argnums=0)
    res_jac_lam = jax.jacfwd(residual, argnums=1)

    def newton_correct(
        z_t: Array, lam: Array, tangent: Array, target_ds: Array
    ) -> tuple[Array, Array, Array]:
        """Correct onto {H = 0, arclength constraint} by Newton on (z, λ)."""
        anchor_z, anchor_lam = z_t, lam
        for _ in range(50):
            h_val = residual(z_t, lam)
            arc = tangent[:dim] @ (z_t - anchor_z) + tangent[dim] * (lam - anchor_lam) - target_ds
            f_val = jnp.concatenate([h_val, jnp.array([arc])])
            if float(jnp.max(jnp.abs(h_val))) < newton_tol and abs(float(arc)) < newton_tol:
                break
            jac_top = jnp.concatenate([res_jac_z(z_t, lam), res_jac_lam(z_t, lam)[:, None]], axis=1)
            jac = jnp.concatenate([jac_top, tangent[None, :]], axis=0)
            delta = jnp.linalg.solve(jac, -f_val)
            z_t = z_t + delta[:dim]
            lam = lam + delta[dim]
        return z_t, lam, jnp.max(jnp.abs(residual(z_t, lam)))

    def branch_tangent(z_t: Array, lam: Array, previous: Array | None) -> Array:
        jac_top = jnp.concatenate([res_jac_z(z_t, lam), res_jac_lam(z_t, lam)[:, None]], axis=1)
        # Nullspace of the (dim x dim+1) Jacobian: smallest right singular vector.
        _, _, vt = jnp.linalg.svd(jac_top)
        t_vec = vt[-1]
        if previous is not None and float(t_vec @ previous) < 0:
            t_vec = -t_vec
        elif previous is None and float(t_vec[dim]) < 0:
            t_vec = -t_vec  # start moving toward increasing λ
        return jnp.asarray(t_vec / jnp.linalg.norm(t_vec))

    z_t = jnp.zeros(dim)  # centroid at λ = 0 (uniform σ)
    lam = jnp.asarray(0.0)
    tangent = branch_tangent(z_t, lam, None)
    step = first_step

    lam_list, sig_list, rho_list, turn_list = [], [], [], []
    prev_dlam = 1.0
    for _ in range(n_points):
        predictor_z = z_t + step * tangent[:dim]
        predictor_lam = lam + step * tangent[dim]
        z_new, lam_new, res = newton_correct(predictor_z, predictor_lam, tangent, jnp.asarray(0.0))
        if float(res) > newton_tol * 100:
            step *= 0.5
            if step < first_step * 1e-6:
                break
            continue
        z_t, lam = z_new, lam_new
        tangent = branch_tangent(z_t, lam, tangent)
        step = min(step * 1.1, first_step * 10)

        sigma = _sigma_from_z(z_t, q_basis, num_actions)
        point = QREPoint(
            sigma=sigma,
            lam=jnp.full((game.n_players,), jnp.maximum(lam, 0.0)),
            expected_payoffs=expected_payoffs(game, sigma),
            residual=res,
            n_iter=jnp.asarray(0),
            payoff_range=payoff_range(game.payoffs),
            converged=jnp.asarray(True),
        )
        ops = build_operators(game, point)
        rho = jnp.max(jnp.abs(jnp.linalg.eigvals(ops.s_tangent @ ops.b_tangent)))

        dlam = float(tangent[dim])
        lam_list.append(float(lam))
        sig_list.append(jnp.concatenate(sigma))
        rho_list.append(float(rho))
        turn_list.append(prev_dlam * dlam < 0)  # λ-direction reversal = fold
        prev_dlam = dlam
        if float(lam) > lam_max:
            break

    return Branch(
        lambdas=jnp.asarray(lam_list),
        sigmas=jnp.stack(sig_list),
        rhos=jnp.asarray(rho_list),
        turning_points=jnp.asarray(turn_list),
    )
