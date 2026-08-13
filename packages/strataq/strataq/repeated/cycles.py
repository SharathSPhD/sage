"""Edgeworth price cycles from logit best-response dynamics on a ladder.

The stage game is homogeneous-good Bertrand on a discrete price ladder: the
cheapest firm takes the market, ties split it, everyone else sells nothing. Under
*simultaneous* logit response that game has a quantal equilibrium and the
dynamics settle. Under *alternating* response — firms move in turn, each replying
to the rival's current mix — the undercutting incentive has nowhere to rest: each
firm shaves the rival's price until the margin is gone, then one relents and
jumps back up. That sawtooth is the Edgeworth cycle, and it is a live research
object here because the same ladder plus the same λ is what the pricing domain
uses.

λ is the only knob between the two regimes: large λ is the classic myopic
undercutting cycle, small λ smooths it into a fixed point.

References
----------
Edgeworth 1925; Maskin–Tirole, Econometrica 1988 (alternating-move price cycles);
Noel, RESTAT 2007 (retail gasoline evidence). Tier: derived — the cycle is a
property of these dynamics, not a theorem about equilibrium.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from strataq.core.defaults import base_config
from strataq.finite.games.tensor import DenseTensorGame, expected_payoffs

__all__ = [
    "PriceCycle",
    "alternating_logit_path",
    "bertrand_ladder",
    "detect_cycle",
    "edgeworth_cycle",
    "linear_market_demand",
]

DemandCurve = Callable[[Array], Array]


def linear_market_demand(intercept: float, slope: float) -> DemandCurve:
    """``D(p) = max(0, intercept - slope p)`` as an elementwise callable."""
    if not float(slope) > 0:
        raise ValueError(f"slope must be > 0, got {slope}")
    a = float(intercept)
    b = float(slope)

    def demand(price: Array) -> Array:
        return jnp.maximum(a - b * price, 0.0)

    return demand


def bertrand_ladder(
    costs: Sequence[float] | Array,
    ladder: Sequence[float] | Array,
    demand: DemandCurve,
    *,
    capacities: Sequence[float] | Array | None = None,
) -> DenseTensorGame:
    """Homogeneous-good Bertrand on a discrete price ladder, with capacities.

    Rationing is efficient: firms strictly cheaper than ``i`` serve up to their
    capacity, ``i`` gets what is left of ``D(p_i)``, and firms at the same price
    split that residual equally, each still capped by its own capacity.

    ``capacities=None`` means unlimited, which collapses to textbook Bertrand —
    the cheapest firm takes the whole market and everyone else earns nothing.
    That case has no cycle: undercutting stops one tick above cost, because
    matching and splitting beats undercutting to a zero margin. **Capacity is
    what makes the cycle**, and it is what Edgeworth's original argument was
    about: a firm that cannot serve the whole market leaves a residual worth
    charging the monopoly price for, so somebody always relents.
    """
    marginal = jnp.asarray(costs, dtype=jnp.float64).ravel()
    levels = jnp.asarray(ladder, dtype=jnp.float64).ravel()
    n_firms = int(marginal.shape[0])
    n_levels = int(levels.shape[0])
    if n_firms < 2:
        raise ValueError(f"a price cycle needs at least two firms, got {n_firms}")
    if n_levels < 2:
        raise ValueError(f"the ladder needs at least two levels, got {n_levels}")
    if not bool(jnp.all(jnp.diff(levels) > 0)):
        raise ValueError("ladder must be strictly increasing")
    if capacities is None:
        limit = jnp.full((n_firms,), float(jnp.max(demand(levels))) * n_firms + 1.0)
    else:
        limit = jnp.asarray(capacities, dtype=jnp.float64).ravel()
        if limit.shape != (n_firms,):
            raise ValueError(f"capacities must have {n_firms} entries, got {int(limit.shape[0])}")
        if not bool(jnp.all(limit > 0)):
            raise ValueError("capacities must be positive")
    mesh = jnp.stack(jnp.meshgrid(*([levels] * n_firms), indexing="ij"), axis=-1)
    shape = mesh.shape[:-1]
    flat = mesh.reshape(-1, n_firms)
    cheaper = flat[:, None, :] < flat[:, :, None]
    served_before = jnp.sum(cheaper * limit[None, None, :], axis=2)
    ties = jnp.sum(flat[:, None, :] == flat[:, :, None], axis=2)
    residual = jnp.maximum(demand(flat) - served_before, 0.0)
    quantity = jnp.minimum(limit[None, :], residual / ties)
    profit = (flat - marginal[None, :]) * quantity
    return DenseTensorGame(tuple(profit[:, i].reshape(shape) for i in range(n_firms)))


def alternating_logit_path(
    game: DenseTensorGame,
    lam: float,
    *,
    n_steps: int | None = None,
    init: Sequence[Array] | None = None,
) -> Array:
    """Round-robin logit best response: one firm revises per step.

    Returns ``(n_steps + 1, n_players, n_levels)`` — the whole trajectory of mixed
    strategies, not just its limit, because the limit is the thing that may not
    exist. All players must share an action count (a common ladder).
    """
    levels = game.num_actions
    if len(set(levels)) != 1:
        raise ValueError(f"alternating dynamics need a shared ladder, got {levels}")
    steps = int(base_config().repeated.cycle_max_steps if n_steps is None else n_steps)
    if steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {steps}")
    precision = float(lam)
    if precision < 0:
        raise ValueError(f"lam must be >= 0, got {precision}")
    n_players = game.n_players
    sigma = (
        [jnp.asarray(s, dtype=jnp.float64) for s in init]
        if init is not None
        else [jnp.full((m,), 1.0 / m, dtype=jnp.float64) for m in levels]
    )
    history = [jnp.stack(sigma)]
    for step in range(steps):
        mover = step % n_players
        utilities = expected_payoffs(game, sigma)
        sigma[mover] = jnp.exp(jax.nn.log_softmax(precision * utilities[mover]))
        history.append(jnp.stack(sigma))
    return jnp.stack(history)


def detect_cycle(series: Array, *, tol: float | None = None) -> tuple[int, int]:
    """Smallest period whose last two repeats match, and where that window starts.

    Returns ``(period, start)`` with ``period = 0`` when the tail of ``series``
    does not repeat — which is the honest answer for a trajectory still in
    transient, not a claim that no cycle exists.
    """
    threshold = base_config().repeated.cycle_tol if tol is None else float(tol)
    length = int(series.shape[0])
    flat = series.reshape(length, -1)
    for period in range(1, length // 2 + 1):
        window = flat[length - period :]
        prior = flat[length - 2 * period : length - period]
        if float(jnp.max(jnp.abs(window - prior))) < threshold:
            return period, length - period
    return 0, length


class PriceCycle(eqx.Module):
    """What the alternating logit dynamics did on the ladder."""

    period: Array
    """Length of the detected cycle in revision steps; 0 if none was detected."""
    amplitude: Array
    """Peak minus trough of the expected market price over the cycle."""
    peak: Array
    trough: Array
    mean_price: Array
    """Time-average expected market price over the cycle."""
    price_path: Array
    """Expected price per firm at every step, ``(n_steps + 1, n_players)``."""
    distribution_path: Array
    """Mixed strategies at every step, ``(n_steps + 1, n_players, n_levels)``."""
    ladder: Array
    lam: Array
    is_fixed_point: Array
    """True when the dynamics settled instead of cycling."""

    @property
    def period_rounds(self) -> float:
        """The period in full rounds of revision (``nan`` when it is not a whole number)."""
        n_players = int(self.price_path.shape[1])
        step = int(self.period)
        if step == 0 or step % n_players:
            return float("nan")
        return float(step // n_players)


def edgeworth_cycle(
    costs: Sequence[float] | Array,
    ladder: Sequence[float] | Array,
    demand: DemandCurve,
    lam: float,
    *,
    capacities: Sequence[float] | Array | None = None,
    n_steps: int | None = None,
    tol: float | None = None,
) -> PriceCycle:
    """Run the ladder dynamics and measure the cycle.

    Amplitude and period are read off the *detected* cycle window, so a
    trajectory that has not repeated yet reports ``period = 0`` and measures the
    whole tail instead of inventing a cycle.
    """
    game = bertrand_ladder(costs, ladder, demand, capacities=capacities)
    levels = jnp.asarray(ladder, dtype=jnp.float64).ravel()
    path = alternating_logit_path(game, lam, n_steps=n_steps)
    prices = path @ levels  # (n_steps + 1, n_players)
    period, start = detect_cycle(path, tol=tol)
    window = prices[start:] if period else prices[prices.shape[0] // 2 :]
    market = jnp.min(window, axis=1)
    peak = jnp.max(market)
    trough = jnp.min(market)
    threshold = base_config().repeated.cycle_tol if tol is None else float(tol)
    return PriceCycle(
        period=jnp.asarray(period, dtype=jnp.int32),
        amplitude=peak - trough,
        peak=peak,
        trough=trough,
        mean_price=jnp.mean(market),
        price_path=prices,
        distribution_path=path,
        ladder=levels,
        lam=jnp.asarray(float(lam)),
        is_fixed_point=jnp.asarray(bool(period == 1 or float(peak - trough) < threshold)),
    )
