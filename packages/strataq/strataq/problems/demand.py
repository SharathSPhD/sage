"""Demand systems for :class:`~strataq.problems.pricing.PricingProblem`.

Three models cover the usual cases: :class:`LogitDemand` (differentiated
products with an outside good), :class:`LinearDemand` (a linear system with own
and cross slopes) and :class:`CustomDemand` (your own ``prices -> quantities``
function). Every model exposes the same two calls — ``quantities(prices)`` and
``elasticities(prices)`` — so a pricing problem never needs to know which one it
was handed.

Elasticities are central differences on ``quantities`` in float64, which keeps
them exact to ~1e-9 for any model, including a ``CustomDemand`` that is not
differentiable by JAX.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["CustomDemand", "DemandModel", "LinearDemand", "LogitDemand"]

_STEP = 1e-6


class DemandModel:
    """Interface: quantities as a function of the price vector."""

    @property
    def n_products(self) -> int:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__

    def quantities(self, prices: Array) -> Array:
        """Quantity demanded of each product at ``prices`` (shape ``(n_products,)``)."""
        raise NotImplementedError

    def bind(self, n_products: int) -> None:
        """Check (or, for :class:`CustomDemand`, fix) the number of products."""
        if self.n_products != n_products:
            raise ValueError(
                f"{self.name} covers {self.n_products} products but the problem has "
                f"{n_products}; give one quality/intercept entry per firm."
            )

    def elasticities(self, prices: Array) -> Array:
        """``E[i, j] = (dq_i/dp_j)(p_j/q_i)`` — own on the diagonal, cross off it."""
        price = jnp.asarray(prices, dtype=jnp.float64).ravel()
        quantity = self.quantities(price)
        step = _STEP * jnp.maximum(jnp.abs(price), 1.0)
        columns = []
        for j in range(int(price.shape[0])):
            bump = jnp.zeros_like(price).at[j].set(step[j])
            columns.append(self.quantities(price + bump) - self.quantities(price - bump))
        jacobian = jnp.stack(columns, axis=1) / (2.0 * step)[None, :]
        safe_q = jnp.where(quantity > 0, quantity, jnp.nan)
        return jacobian * price[None, :] / safe_q[:, None]


class LogitDemand(DemandModel):
    """Multinomial-logit shares: ``u_i = quality_i - price_sensitivity * p_i``.

    With ``outside_option=True`` (the default) the outside good has utility 0, so
    shares never sum to one and a monopolist faces a finite elasticity.
    """

    def __init__(
        self,
        price_sensitivity: float,
        quality: Sequence[float] | Array,
        *,
        market_size: float = 1.0,
        outside_option: bool = True,
    ) -> None:
        beta = float(price_sensitivity)
        if not beta > 0:
            raise ValueError(f"price_sensitivity must be > 0, got {beta}")
        levels = jnp.asarray(quality, dtype=jnp.float64).ravel()
        if levels.shape[0] < 1:
            raise ValueError("quality needs one entry per product")
        if not bool(jnp.all(jnp.isfinite(levels))):
            raise ValueError("quality must be finite")
        if not float(market_size) > 0:
            raise ValueError(f"market_size must be > 0, got {market_size}")
        self.price_sensitivity = beta
        self.quality = levels
        self.market_size = float(market_size)
        self.outside_option = bool(outside_option)
        self._n = int(levels.shape[0])

    @property
    def n_products(self) -> int:
        return self._n

    def quantities(self, prices: Array) -> Array:
        utility = self.quality - self.price_sensitivity * jnp.asarray(prices, dtype=jnp.float64)
        if self.outside_option:
            utility = jnp.concatenate([utility, jnp.zeros(1, dtype=jnp.float64)])
        shares = jax.nn.softmax(utility)
        return self.market_size * shares[: self._n]


class LinearDemand(DemandModel):
    """``q_i = intercept_i - own_slope * p_i + cross_slope * sum_{j != i} p_j``, floored at 0."""

    def __init__(
        self,
        intercept: Sequence[float] | Array,
        own_slope: float,
        cross_slope: float = 0.0,
    ) -> None:
        levels = jnp.asarray(intercept, dtype=jnp.float64).ravel()
        if levels.shape[0] < 1:
            raise ValueError("intercept needs one entry per product")
        if not bool(jnp.all(jnp.isfinite(levels))):
            raise ValueError("intercept must be finite")
        own = float(own_slope)
        cross = float(cross_slope)
        if not own > 0:
            raise ValueError(f"own_slope must be > 0, got {own}")
        if cross < 0:
            raise ValueError(f"cross_slope must be >= 0, got {cross}")
        if levels.shape[0] > 1 and cross >= own:
            raise ValueError(
                f"cross_slope {cross} must be < own_slope {own}: the demand system is "
                "otherwise not downward sloping in own price."
            )
        self.intercept = levels
        self.own_slope = own
        self.cross_slope = cross
        self._n = int(levels.shape[0])

    @property
    def n_products(self) -> int:
        return self._n

    def quantities(self, prices: Array) -> Array:
        price = jnp.asarray(prices, dtype=jnp.float64)
        rivals = jnp.sum(price) - price
        return jnp.maximum(self.intercept - self.own_slope * price + self.cross_slope * rivals, 0.0)


class CustomDemand(DemandModel):
    """Your own ``prices -> quantities`` callable.

    ``n_products`` may be left unset, in which case the pricing problem fixes it
    from the number of firms and checks the callable returns a matching vector.
    """

    def __init__(
        self,
        fn: Callable[[Array], Array],
        *,
        n_products: int | None = None,
        name: str = "CustomDemand",
    ) -> None:
        if not callable(fn):
            raise ValueError("CustomDemand needs a callable mapping prices to quantities")
        self.fn = fn
        self._n = n_products
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_products(self) -> int:
        if self._n is None:
            raise ValueError("CustomDemand n_products is unset; pass n_products=... ")
        return self._n

    def bind(self, n_products: int) -> None:
        if self._n is None:
            self._n = n_products
        elif self._n != n_products:
            raise ValueError(
                f"CustomDemand covers {self._n} products but the problem has {n_products}"
            )
        probe = self.quantities(jnp.ones((n_products,), dtype=jnp.float64))
        if probe.shape != (n_products,):
            raise ValueError(
                f"CustomDemand callable returned shape {probe.shape}, expected ({n_products},)"
            )

    def quantities(self, prices: Array) -> Array:
        out = jnp.asarray(self.fn(jnp.asarray(prices, dtype=jnp.float64)), dtype=jnp.float64)
        return out.ravel()
