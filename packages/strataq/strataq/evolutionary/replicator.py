"""Replicator dynamics on a single population, continuous and discrete.

The state is a share vector ``x`` on the simplex and the game is a symmetric
payoff matrix ``A``, so type ``i`` earns ``(Ax)_i`` against the population.
Continuous replicator is ``ẋ_i = x_i((Ax)_i − x·Ax)``; the discrete map is
``x'_i = x_i (w + (Ax)_i) / (w + x·Ax)`` with a background fitness ``w`` that
keeps the numerator positive.

Rest points are found by support enumeration — exactly, by linear solve on each
support, not by integrating and hoping — and classified by the eigenvalues of the
field's Jacobian projected onto the simplex tangent space. At a vertex those
eigenvalues are the invasion fitnesses ``A_ik − A_kk``, which is the check the
tests use.

The *logit dynamic* lives here too, because its rest points are the object this
project cares about: ``x = softmax(λ A x)`` is exactly the symmetric logit QRE of
the same game at precision λ, so the evolutionary and the finite readings of one
game meet at one number.

References
----------
Taylor–Jonker 1978 (replicator); Maynard Smith 1982; Weibull 1995 §3–4;
Sandholm 2010 §5.5 and §6.2 (logit dynamic and its QRE rest points).
Tier: exact.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from strataq.core.defaults import base_config

__all__ = [
    "CENTRE",
    "DEGENERATE",
    "SADDLE",
    "STABLE",
    "UNSTABLE",
    "RestPoint",
    "discrete_replicator",
    "logit_dynamic_field",
    "logit_rest_point",
    "replicator_field",
    "replicator_flow",
    "rest_points",
    "stability",
    "tangent_basis",
]

STABLE = "stable"
UNSTABLE = "unstable"
SADDLE = "saddle"
CENTRE = "centre"
DEGENERATE = "degenerate"


def _matrix(payoff: Sequence[Sequence[float]] | Array) -> Array:
    a = jnp.asarray(payoff, dtype=jnp.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"a symmetric game needs a square payoff matrix, got {tuple(a.shape)}")
    if a.shape[0] < 2:
        raise ValueError("a symmetric game needs at least two types")
    if not bool(jnp.all(jnp.isfinite(a))):
        raise ValueError("payoff matrix must be finite")
    return a


def _share(x: Sequence[float] | Array, n: int) -> Array:
    share = jnp.asarray(x, dtype=jnp.float64).ravel()
    if share.shape != (n,):
        raise ValueError(f"state must have {n} entries, got {int(share.shape[0])}")
    if bool(jnp.any(share < 0.0)):
        raise ValueError("state must be non-negative")
    total = float(jnp.sum(share))
    if not abs(total - 1.0) < 1e-8:
        raise ValueError(f"state must sum to 1, got {total}")
    return share


def replicator_field(payoff: Sequence[Sequence[float]] | Array, x: Array) -> Array:
    """``ẋ_i = x_i((Ax)_i − x·Ax)`` — the continuous replicator vector field."""
    a = _matrix(payoff)
    share = _share(x, int(a.shape[0]))
    fitness = a @ share
    return share * (fitness - share @ fitness)


def replicator_flow(
    payoff: Sequence[Sequence[float]] | Array,
    x0: Sequence[float] | Array,
    *,
    step: float | None = None,
    steps: int | None = None,
) -> Array:
    """Integrate the continuous replicator with RK4, returning the whole trajectory.

    Shape ``(steps + 1, n_types)``. The simplex is invariant under the exact flow;
    RK4 preserves it to ``O(step^4)`` per step and the result is renormalised each
    step so the state stays a distribution.
    """
    cfg = base_config().evolutionary
    dt = float(cfg.step if step is None else step)
    horizon = int(cfg.steps if steps is None else steps)
    if dt <= 0:
        raise ValueError(f"step must be > 0, got {dt}")
    if horizon < 1:
        raise ValueError(f"steps must be >= 1, got {horizon}")
    a = _matrix(payoff)
    state = _share(x0, int(a.shape[0]))

    def field(share: Array) -> Array:
        fitness = a @ share
        return share * (fitness - share @ fitness)

    def advance(share: Array, _: Array) -> tuple[Array, Array]:
        k1 = field(share)
        k2 = field(share + 0.5 * dt * k1)
        k3 = field(share + 0.5 * dt * k2)
        k4 = field(share + dt * k3)
        nxt = share + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        nxt = jnp.clip(nxt, 0.0, jnp.inf)
        nxt = nxt / jnp.sum(nxt)
        return nxt, nxt

    _, path = jax.lax.scan(advance, state, jnp.arange(horizon))
    return jnp.concatenate([state[None, :], path], axis=0)


def discrete_replicator(
    payoff: Sequence[Sequence[float]] | Array,
    x0: Sequence[float] | Array,
    *,
    steps: int,
    background: float = 0.0,
) -> Array:
    """Iterate ``x'_i = x_i (w + (Ax)_i) / (w + x·Ax)``, returning the trajectory.

    ``background`` is the ``w`` that keeps fitness positive when payoffs can be
    negative; with ``w = 0`` and non-negative payoffs this is the textbook
    discrete replicator. Raises when a fitness goes non-positive rather than
    silently producing a negative share.
    """
    a = _matrix(payoff)
    state = _share(x0, int(a.shape[0]))
    if int(steps) < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    w = float(background)
    fitness = w + a @ state
    if bool(jnp.any(fitness < 0.0)):
        raise ValueError(
            "discrete replicator needs non-negative fitness; raise background= above "
            f"{-float(jnp.min(a @ state)):.4g}"
        )

    def advance(share: Array, _: Array) -> tuple[Array, Array]:
        gains = w + a @ share
        mean = share @ gains
        nxt = share * gains / mean
        return nxt, nxt

    _, path = jax.lax.scan(advance, state, jnp.arange(int(steps)))
    return jnp.concatenate([state[None, :], path], axis=0)


def tangent_basis(n: int) -> Array:
    """An orthonormal basis of ``{v ∈ R^n : Σv = 0}``, shape ``(n, n-1)``."""
    projector = jnp.eye(n) - jnp.full((n, n), 1.0 / n)
    u, s, _ = jnp.linalg.svd(projector)
    keep = s > 0.5
    return u[:, keep]


class RestPoint(eqx.Module):
    """A rest point of the replicator, with its stability read-out."""

    x: Array
    """The share vector, ``(n_types,)``."""
    support: Array
    """Boolean mask of the types present."""
    eigenvalues: Array
    """Spectrum of the Jacobian projected onto the simplex tangent space."""
    kind: str = eqx.field(static=True)
    """``stable`` / ``unstable`` / ``saddle`` / ``centre`` / ``degenerate``."""
    is_nash: bool = eqx.field(static=True)
    """True when no absent type has strictly higher fitness — the Nash condition."""

    @property
    def max_real_part(self) -> float:
        return float(jnp.max(jnp.real(self.eigenvalues)))


def stability(
    payoff: Sequence[Sequence[float]] | Array,
    x: Sequence[float] | Array,
    *,
    tol: float | None = None,
) -> tuple[Array, str]:
    """Tangent-space eigenvalues at ``x`` and their classification.

    At a vertex ``e_k`` this returns exactly the invasion fitnesses
    ``A_ik − A_kk`` for ``i ≠ k``; in the interior it is the Jacobian of the
    replicator restricted to the simplex.
    """
    a = _matrix(payoff)
    n = int(a.shape[0])
    state = _share(x, n)
    threshold = base_config().evolutionary.rest_tol if tol is None else float(tol)

    def field(share: Array) -> Array:
        fitness = a @ share
        return share * (fitness - share @ fitness)

    jacobian = jax.jacobian(field)(state)
    basis = tangent_basis(n)
    eigenvalues = jnp.linalg.eigvals(basis.T @ jacobian @ basis)
    real = jnp.real(eigenvalues)
    positive = bool(jnp.any(real > threshold))
    negative = bool(jnp.any(real < -threshold))
    if positive and negative:
        kind = SADDLE
    elif positive:
        kind = UNSTABLE
    elif negative:
        kind = STABLE
    elif bool(jnp.all(jnp.abs(real) <= threshold)) and bool(
        jnp.any(jnp.abs(jnp.imag(eigenvalues)) > threshold)
    ):
        kind = CENTRE
    else:
        kind = DEGENERATE
    return eigenvalues, kind


def rest_points(
    payoff: Sequence[Sequence[float]] | Array, *, tol: float | None = None
) -> tuple[RestPoint, ...]:
    """Every rest point of the replicator, by support enumeration.

    On support ``S`` a rest point solves ``(Ax)_i = c`` for all ``i ∈ S`` with
    ``Σ_S x = 1`` and ``x_S > 0`` — one small linear system per support, so the
    answer is exact and complete rather than whatever an integrator wandered
    into. Cost is ``2^n`` solves; ``n`` is the number of types, not the
    population.
    """
    a = _matrix(payoff)
    n = int(a.shape[0])
    threshold = base_config().evolutionary.rest_tol if tol is None else float(tol)
    found: list[RestPoint] = []
    seen: list[Array] = []
    for size in range(1, n + 1):
        for support in itertools.combinations(range(n), size):
            block = a[jnp.asarray(support)][:, jnp.asarray(support)]
            system = jnp.zeros((size + 1, size + 1), dtype=jnp.float64)
            system = system.at[:size, :size].set(block)
            system = system.at[:size, size].set(-1.0)
            system = system.at[size, :size].set(1.0)
            rhs = jnp.zeros((size + 1,), dtype=jnp.float64).at[size].set(1.0)
            if float(jnp.abs(jnp.linalg.det(system))) < threshold:
                continue
            solution = jnp.linalg.solve(system, rhs)[:size]
            if bool(jnp.any(solution <= threshold)):
                continue
            x = jnp.zeros((n,), dtype=jnp.float64).at[jnp.asarray(support)].set(solution)
            if any(float(jnp.max(jnp.abs(x - prior))) < 1e-9 for prior in seen):
                continue
            seen.append(x)
            fitness = a @ x
            eigenvalues, kind = stability(a, x, tol=threshold)
            found.append(
                RestPoint(
                    x=x,
                    support=x > threshold,
                    eigenvalues=eigenvalues,
                    kind=kind,
                    is_nash=bool(jnp.max(fitness) <= float(x @ fitness) + 1e-9),
                )
            )
    return tuple(found)


def logit_dynamic_field(
    payoff: Sequence[Sequence[float]] | Array, x: Sequence[float] | Array, lam: float
) -> Array:
    """``ẋ = softmax(λ A x) − x`` — the logit (perturbed best response) dynamic.

    Its rest points are the symmetric logit QRE of the same game at precision λ,
    which is the bridge between this module and :mod:`strataq.finite`.
    """
    a = _matrix(payoff)
    state = _share(x, int(a.shape[0]))
    return jnp.exp(jax.nn.log_softmax(float(lam) * (a @ state))) - state


def logit_rest_point(
    payoff: Sequence[Sequence[float]] | Array,
    lam: float,
    *,
    x0: Sequence[float] | Array | None = None,
    damping: float | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
) -> Array:
    """The rest point of the logit dynamic: ``x = softmax(λ A x)``, by damped iteration."""
    a = _matrix(payoff)
    n = int(a.shape[0])
    cfg = base_config()
    damp = cfg.solver.damping if damping is None else float(damping)
    threshold = cfg.tolerances.solve if tol is None else float(tol)
    limit = cfg.solver.max_iter if max_iter is None else int(max_iter)
    precision = float(lam)
    if precision < 0:
        raise ValueError(f"lam must be >= 0, got {precision}")
    state = jnp.full((n,), 1.0 / n) if x0 is None else _share(x0, n)

    def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        share, it, _ = carry
        target = jnp.exp(jax.nn.log_softmax(precision * (a @ share)))
        residual = jnp.max(jnp.abs(target - share))
        return ((1.0 - damp) * share + damp * target, it + 1, residual)

    def cond(carry: tuple[Array, Array, Array]) -> Array:
        _, it, residual = carry
        return (residual >= threshold) & (it < limit)

    final: tuple[Array, Array, Array] = jax.lax.while_loop(
        cond, body, (state, jnp.asarray(0), jnp.asarray(jnp.inf))
    )
    return final[0]
