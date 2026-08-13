"""Agent QRE: logit at every information set, on the same damped fixed point.

The agent normal form treats each information set as its own decision maker.
That agent chooses with a logit rule over the expected payoff *conditional on
reaching* the information set, with beliefs over the nodes in it given by their
relative reach probabilities:

``σ_h(a) ∝ exp(λ Σ_{x∈h} μ(x) V_{i(h)}(x → a))``,  ``μ(x) = ρ(x) / Σ_{y∈h} ρ(y)``

This is McKelvey–Palfrey (1998) and it is the thing Gambit is known for. Solved
by the same damped Krasnoselskii–Mann iteration the strategic-form solver uses,
with the two tree passes inside: one forward for ``ρ``, one backward for ``V``.

At λ → 0 every agent randomises uniformly; as λ → ∞ the branch through the
solution converges to a sequential equilibrium. What it does *not* do is
back-induct: on the centipede game the AQRE keeps play alive for many moves at
precisions where backward induction says stop immediately, which is the empirical
point of the model.

References
----------
McKelvey–Palfrey, GEB 1995 (logit QRE); McKelvey–Palfrey, Experimental Economics
1998 (agent QRE for extensive games). Tier: exact — the definition is theirs and
the solver is cross-checked against Gambit's homotopy.
"""

from __future__ import annotations

import sys

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from strataq.core.defaults import base_config
from strataq.extensive.behaviour import (
    node_values,
    policy_from_behaviour,
    reach_probabilities,
    uniform_behaviour,
)
from strataq.extensive.tree import ExtensiveGame

__all__ = ["AQREPoint", "agent_qre", "agent_qre_branch"]

_TINY = sys.float_info.min


class AQREPoint(eqx.Module):
    """A solved agent quantal response equilibrium."""

    behaviour: Array
    """``(n_infosets, max_actions)``: the logit choice at each information set."""
    utilities: Array
    """``(n_infosets, max_actions)``: the conditional payoffs the logit acted on."""
    values: Array
    """``(n_nodes, n_players)``: expected payoff to each player from each node."""
    reach: Array
    """``(n_nodes,)``: probability of reaching each node."""
    lam: Array
    residual: Array
    n_iter: Array
    converged: Array

    @property
    def expected_payoffs(self) -> Array:
        """Expected payoff to each player at the root, ``(n_players,)``."""
        return self.values[0]

    @property
    def beliefs(self) -> Array:
        """``(n_nodes,)``: the belief weight of each node within its information set."""
        return self.reach


def _responder(game: ExtensiveGame):  # type: ignore[no-untyped-def]
    """Build the one-sweep logit response map for this tree (structure closed over)."""
    decision = jnp.flatnonzero(game.is_decision)
    owners = game.infoset_player[game.infoset[decision]]
    kids = game.children[decision]
    valid = kids >= 0
    safe_kids = jnp.where(valid, kids, 0)
    targets = game.infoset[decision]
    mask = game.action_mask()
    penalty = jnp.where(mask, 0.0, -jnp.inf)
    # The depth groupings need concrete values, so they are computed once, here,
    # and never inside the traced fixed-point body.
    levels = game.levels()
    internal = game.internal_levels()

    def respond(behaviour: Array, lam: Array) -> tuple[Array, Array, Array, Array]:
        policy = policy_from_behaviour(game, behaviour)
        reach = reach_probabilities(game, policy, levels=levels)
        values = node_values(game, policy, internal=internal)
        child_values = values[safe_kids, owners[:, None]]
        weight = reach[decision][:, None]
        contribution = jnp.where(valid, weight * child_values, 0.0)
        totals = jnp.zeros((game.n_infosets, game.max_actions), dtype=jnp.float64)
        totals = totals.at[targets].add(contribution)
        mass = jnp.zeros((game.n_infosets,), dtype=jnp.float64).at[targets].add(reach[decision])
        utilities = totals / jnp.maximum(mass, _TINY)[:, None]
        target = jnp.exp(jax.nn.log_softmax(lam * utilities + penalty, axis=-1))
        return jnp.where(mask, target, 0.0), utilities, values, reach

    return respond


class _Solver:
    """The compiled damped iteration for one tree.

    Built once and reused for every damping and every λ in a solve. That matters:
    the continuation fallback makes hundreds of runs, and tracing the
    ``while_loop`` afresh each time costs far more than the arithmetic does.
    """

    def __init__(self, game: ExtensiveGame) -> None:
        self.game = game
        self.respond = _responder(game)

        def run(
            start: Array, lam: Array, damp: Array, tol: Array, limit: Array
        ) -> tuple[Array, Array, Array]:
            def cond(state: tuple[Array, Array, Array]) -> Array:
                _, it, residual = state
                return (residual >= tol) & (it < limit)

            def body(state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
                behaviour, it, _ = state
                target, _, _, _ = self.respond(behaviour, lam)
                residual = jnp.max(jnp.abs(target - behaviour))
                return ((1.0 - damp) * behaviour + damp * target, it + 1, residual)

            solved: tuple[Array, Array, Array] = jax.lax.while_loop(
                cond, body, (start, jnp.asarray(0), jnp.asarray(jnp.inf))
            )
            return solved

        self.run = jax.jit(run)

    def attempt(
        self, lam: float, start: Array, damping: float, tol: float, max_iter: int
    ) -> AQREPoint:
        """One damped Krasnoselskii–Mann run at a fixed damping."""
        lam_array = jnp.asarray(float(lam))
        behaviour, n_iter, residual = self.run(
            start,
            lam_array,
            jnp.asarray(float(damping)),
            jnp.asarray(float(tol)),
            jnp.asarray(int(max_iter)),
        )
        _, utilities, values, reach = self.respond(behaviour, lam_array)
        return AQREPoint(
            behaviour=behaviour,
            utilities=utilities,
            values=values,
            reach=reach,
            lam=lam_array,
            residual=residual,
            n_iter=n_iter,
            converged=residual < float(tol),
        )

    def with_backoff(
        self,
        lam: float,
        start: Array,
        damping: float,
        tol: float,
        max_iter: int,
        restarts: int,
        backoff: float,
    ) -> AQREPoint:
        """Damped iteration, backing the damping off until it stops cycling."""
        damp = damping
        point = self.attempt(lam, start, damp, tol, max_iter)
        for _ in range(int(restarts)):
            if bool(point.converged):
                return point
            damp *= backoff
            point = self.attempt(lam, start, damp, tol, max_iter)
        return point


def _settings(
    damping: float | None, tol: float | None, max_iter: int | None
) -> tuple[float, float, int, int, float, int]:
    """Resolve solver settings from ``config/base.yaml``."""
    cfg = base_config()
    return (
        float(cfg.solver.damping if damping is None else damping),
        float(cfg.tolerances.solve if tol is None else tol),
        int(cfg.solver.max_iter if max_iter is None else max_iter),
        int(cfg.extensive.max_restarts),
        float(cfg.extensive.damping_backoff),
        int(cfg.extensive.continuation_points),
    )


def agent_qre(
    game: ExtensiveGame,
    lam: float,
    *,
    init: Array | None = None,
    damping: float | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
) -> AQREPoint:
    """Solve the agent QRE at precision λ.

    Damped iteration first, at the damping from ``config/base.yaml``. The map is
    not a contraction at high precision — on Kuhn poker it cycles above about
    λ = 3 — so non-convergence is met with two escapes, in order: progressively
    smaller damping, then continuation in λ from the uniform profile upward, each
    step of which gets the same backoff. That is Gambit's homotopy idea without
    the corrector, and it is what makes the solver usable at the precisions
    people actually fit.

    A caller that supplies ``init`` gets a single backed-off run and no
    continuation, because the point of supplying a starting profile is to choose
    the branch yourself.
    """
    precision = float(lam)
    if precision < 0:
        raise ValueError(f"lam must be >= 0, got {precision}")
    damp, threshold, limit, restarts, backoff, points = _settings(damping, tol, max_iter)
    solver = _Solver(game)
    start = uniform_behaviour(game) if init is None else jnp.asarray(init, dtype=jnp.float64)

    point = solver.with_backoff(precision, start, damp, threshold, limit, restarts, backoff)
    if bool(point.converged) or init is not None:
        return point

    behaviour = uniform_behaviour(game)
    for value in jnp.linspace(0.0, precision, points):
        point = solver.with_backoff(
            float(value), behaviour, damp, threshold, limit, restarts, backoff
        )
        behaviour = point.behaviour
    return point


def agent_qre_branch(
    game: ExtensiveGame,
    lam_max: float,
    *,
    n_points: int = 40,
    damping: float | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
) -> tuple[Array, Array]:
    """Trace the principal AQRE branch from λ = 0 upward by continuation.

    Each solve starts from the previous one, which is what keeps the trace on one
    branch. Returns ``(lambdas, behaviours)`` with shapes ``(n_points,)`` and
    ``(n_points, n_infosets, max_actions)``.
    """
    if float(lam_max) <= 0:
        raise ValueError(f"lam_max must be > 0, got {lam_max}")
    if int(n_points) < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    damp, threshold, limit, restarts, backoff, _ = _settings(damping, tol, max_iter)
    solver = _Solver(game)
    grid = jnp.linspace(0.0, float(lam_max), int(n_points))
    behaviour = uniform_behaviour(game)
    trace = []
    for value in grid:
        point = solver.with_backoff(
            float(value), behaviour, damp, threshold, limit, restarts, backoff
        )
        behaviour = point.behaviour
        trace.append(behaviour)
    return grid, jnp.stack(trace)
