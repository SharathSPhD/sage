"""strataq.problems: every problem type solved against a known or derivable answer.

The pricing cases are checked against closed-form monopoly optima, the auction
against a single-bidder reserve argument, routing against the committed Sioux
Falls flows and an analytic two-route equilibrium, allocation against a brute
force enumeration of the Blotto grid, and electricity against the exact
distribution of the minimum of two uniform draws.
"""

import math
import warnings

import jax.numpy as jnp
import pytest
import strataq as sq
from strataq.problems import ConvergenceWarning

PRICING_SUMMARY = """                    strataq PricingProblem
==============================================================
price                        1.29   firms                    2
profit                   0.002746   price levels             9
margin                       0.29   precision              1.5
own elasticity               -4.6   demand          LogitDemand
cross elasticity          0.02242   mean rival price      1.49
=============================================================="""

PARALLEL_SUMMARY = """                    strataq RoutingProblem
==============================================================
total cost                  7.999   links                    4
mean travel time            2.666   routes                   2
max volume/capacity         1.665   od pairs                 1
toll revenue                    0   precision              100
cost change                    --   converged             True
=============================================================="""

# Committed Sioux Falls reading: theta = 0.5, top-12 OD pairs, k = 3 routes.
SIOUX_TOTAL_COST = 238177.376545206
SIOUX_LINK_FLOWS = {
    8: 17.426488225,
    9: 17.849639204,
    24: 2813.896676818,
    27: 4229.851384367,
    28: 6026.462498142,
    29: 2091.869575592,
}

PARALLEL_EDGES = [
    (1, 2, 1.0, 1.0, 1.0, 1.0),  # cost 1 + v
    (1, 3, 2.0, 1.0, 0.25, 1.0),  # cost 2 + 0.5 v
    (2, 4, 0.0, 1.0, 0.0, 1.0),
    (3, 4, 0.0, 1.0, 0.0, 1.0),
]


def canonical_pricing() -> sq.PricingProblem:
    return sq.PricingProblem(
        costs=[1.00, 1.05],
        grid=(1.09, 1.89, 0.10),
        demand=sq.LogitDemand(price_sensitivity=3.6, quality=[0.0, -0.1]),
        precision=1.5,
    )


def sioux_falls(**kwargs: object) -> sq.RoutingSolution:
    try:
        problem = sq.RoutingProblem(network="sioux_falls", precision=0.5, max_od=12, **kwargs)
    except OSError:
        pytest.skip("TNTP fetch unavailable (offline)")
    return problem.solve()


class TestPricing:
    def test_linear_monopoly_price_is_exact_on_the_grid(self):
        """(p - c)(a - b p) peaks at p = (a/b + c)/2 = 6, which is a grid level."""
        res = sq.PricingProblem(
            costs=[2.0],
            grid=(2.0, 10.0, 0.5),
            demand=sq.LinearDemand(intercept=[10.0], own_slope=1.0),
            precision=20.0,
        ).solve()
        assert res.success
        assert res.price == pytest.approx(6.0)
        assert res.profit == pytest.approx(16.0)
        assert res.n_firms == 1
        assert res.rival_prices.shape == (0, 17)

    def test_logit_monopoly_price_is_within_one_grid_step(self):
        """The first-order condition p - c = 1 / (beta (1 - share)) solved by bisection."""
        beta, cost = 3.6, 1.0

        def gap(price: float) -> float:
            share = 1.0 / (1.0 + math.exp(beta * price))
            return price - cost - 1.0 / (beta * (1.0 - share))

        low, high = 1.01, 3.0
        for _ in range(200):
            mid = 0.5 * (low + high)
            low, high = (mid, high) if gap(mid) < 0 else (low, mid)
        optimum = 0.5 * (low + high)

        res = sq.PricingProblem(
            costs=[cost],
            grid=(1.05, 2.05, 0.05),
            demand=sq.LogitDemand(price_sensitivity=beta, quality=[0.0]),
            precision=20.0,
        ).solve()
        assert abs(res.price - optimum) <= 0.05

    def test_duopoly_rival_distribution_is_a_distribution(self):
        res = canonical_pricing().solve()
        assert res.success
        assert res.rival_prices.shape == (1, 9)
        assert float(jnp.sum(res.rival_prices)) == pytest.approx(1.0)
        assert bool(jnp.all(res.rival_prices >= 0))
        low, high = float(res.price_grid[0]), float(res.price_grid[-1])
        assert low <= float(res.expected_rival_prices[0]) <= high
        assert res.profit == pytest.approx(float(jnp.max(res.profit_curve)))

    def test_logit_elasticities_match_closed_form(self):
        beta = 3.6
        demand = sq.LogitDemand(price_sensitivity=beta, quality=[0.0, -0.1])
        prices = jnp.asarray([1.30, 1.45])
        shares = demand.quantities(prices)
        elasticities = demand.elasticities(prices)
        for i in range(2):
            own = -beta * float(prices[i]) * (1.0 - float(shares[i]))
            assert float(elasticities[i, i]) == pytest.approx(own, abs=1e-6)
            j = 1 - i
            cross = beta * float(prices[j]) * float(shares[j])
            assert float(elasticities[i, j]) == pytest.approx(cross, abs=1e-6)

    def test_custom_demand_reproduces_the_linear_answer(self):
        linear = sq.LinearDemand(intercept=[10.0, 10.0], own_slope=1.0, cross_slope=0.4)
        custom = sq.CustomDemand(lambda p: jnp.maximum(10.0 - p + 0.4 * (jnp.sum(p) - p), 0.0))
        common = {"costs": [2.0, 2.0], "grid": (2.0, 12.0, 0.5), "precision": 5.0}
        assert (
            sq.PricingProblem(demand=custom, **common).solve().price
            == sq.PricingProblem(demand=linear, **common).solve().price
        )

    def test_summary_is_stable(self):
        assert str(canonical_pricing().solve().summary()) == PRICING_SUMMARY

    def test_physics_is_only_under_diagnostics(self):
        res = canonical_pricing().solve()
        for banned in ("alpha", "epr", "entropy_production", "reciprocity_defect", "rho_sb"):
            assert not hasattr(res, banned), f"{banned} must not sit on the answer"
        diagnostics = res.diagnostics
        assert 0.0 <= diagnostics.alpha <= 1.0
        assert diagnostics.reciprocity_defect >= 0.0
        assert diagnostics.residual < 1e-8

    def test_non_convergence_is_a_flag_and_a_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = sq.PricingProblem(
                costs=[1.00, 1.05],
                grid=(1.09, 1.89, 0.10),
                demand=sq.LogitDemand(price_sensitivity=3.6, quality=[0.0, -0.1]),
                precision=1.5,
                max_iter=1,
            ).solve()
        assert res.success is False
        assert "did not converge" in res.message
        assert any(issubclass(w.category, ConvergenceWarning) for w in caught)

    @pytest.mark.parametrize(
        ("kwargs", "fragment"),
        [
            ({"grid": (2.0, 1.0, 0.5)}, "stop must be >= start"),
            ({"grid": (1.0, 2.0, -0.5)}, "step must be > 0"),
            ({"grid": [1.0]}, "at least 2 levels"),
            ({"precision": 0.0}, "precision must be > 0"),
            ({"firm": 5}, "firm must be in"),
        ],
    )
    def test_bad_input_raises_value_error(self, kwargs, fragment):
        base = {
            "costs": [1.0, 1.0],
            "grid": (1.0, 2.0, 0.25),
            "demand": sq.LogitDemand(price_sensitivity=2.0, quality=[0.0, 0.0]),
        }
        with pytest.raises(ValueError, match=fragment):
            sq.PricingProblem(**{**base, **kwargs})

    def test_demand_width_must_match_the_firms(self):
        with pytest.raises(ValueError, match="one quality/intercept entry per firm"):
            sq.PricingProblem(
                costs=[1.0, 1.0],
                grid=(1.0, 2.0, 0.25),
                demand=sq.LogitDemand(price_sensitivity=2.0, quality=[0.0]),
            )


class TestAuction:
    def test_single_bidder_bids_the_reserve(self):
        """Alone against a reserve, the whole problem is: pay the reserve, keep the rest."""
        res = sq.AuctionProblem(
            values=10.0, n_bidders=1, grid=(4.0, 10.0, 0.5), reserve=4.0, precision=5.0
        ).solve()
        assert res.success
        assert res.bid == pytest.approx(4.0)
        assert res.surplus == pytest.approx(6.0)
        assert res.win_probability == pytest.approx(1.0)
        assert res.expected_clearing_bid == pytest.approx(res.own_bid_distribution @ res.bid_grid)

    def test_single_supplier_offers_the_ceiling(self):
        res = sq.AuctionProblem(
            costs=3.0, n_bidders=1, grid=(3.0, 9.0, 0.5), reserve=9.0, precision=5.0
        ).solve()
        assert res.bid == pytest.approx(9.0)
        assert res.surplus == pytest.approx(6.0)
        assert res.win_probability == pytest.approx(1.0)

    def test_symmetric_bidders_get_a_symmetric_equilibrium(self):
        res = sq.AuctionProblem(
            values=[10.0, 10.0, 10.0], grid=(5.0, 10.0, 0.5), precision=2.0
        ).solve()
        assert res.success
        assert res.n_bidders == 3
        assert res.rival_bids.shape == (2, 11)
        for row in res.rival_bids:
            assert float(jnp.max(jnp.abs(row - res.own_bid_distribution))) < 1e-8
        assert 0.0 <= res.win_probability <= 1.0
        assert res.surplus == pytest.approx(float(jnp.max(res.surplus_curve)))

    def test_bids_below_the_reserve_never_win(self):
        res = sq.AuctionProblem(
            values=[10.0, 10.0], grid=(1.0, 10.0, 1.0), reserve=6.0, precision=1.0
        ).solve()
        below = res.bid_grid < 6.0
        assert float(jnp.max(res.win_curve[below])) == pytest.approx(0.0)

    def test_needs_exactly_one_of_values_or_costs(self):
        with pytest.raises(ValueError, match="exactly one of values"):
            sq.AuctionProblem(grid=(1.0, 2.0, 0.5), values=[1.0], costs=[1.0])
        with pytest.raises(ValueError, match="exactly one of values"):
            sq.AuctionProblem(grid=(1.0, 2.0, 0.5))


class TestRouting:
    def test_sioux_falls_reproduces_the_committed_flows(self):
        res = sioux_falls()
        assert res.success
        assert (res.n_links, res.n_routes, res.n_od) == (76, 36, 12)
        assert res.total_demand == pytest.approx(43700.0)
        assert res.total_cost == pytest.approx(SIOUX_TOTAL_COST, rel=1e-9)
        for link, flow in SIOUX_LINK_FLOWS.items():
            assert float(res.flows[link]) == pytest.approx(flow, rel=1e-8)
        assert res.diagnostics.symmetry_defect < 1e-12

    def test_sioux_falls_toll_moves_flow_off_the_tolled_link(self):
        base = sioux_falls()
        busiest = int(jnp.argmax(base.flows))
        tolled = sioux_falls(tolls={busiest: 5.0})
        assert tolled.toll_effect is not None
        assert tolled.toll_effect.tolled_links == (busiest,)
        assert float(tolled.flows[busiest]) < float(base.flows[busiest])
        assert tolled.toll_effect.revenue == pytest.approx(5.0 * float(tolled.flows[busiest]))
        assert tolled.toll_effect.delta_total_cost == pytest.approx(
            tolled.total_cost - base.total_cost
        )

    def test_two_route_network_matches_the_analytic_equilibrium(self):
        """1 + v0 = 2 + 0.5 v1 with v0 + v1 = 3 gives v0 = 5/3."""
        res = sq.RoutingProblem(
            network=PARALLEL_EDGES, demand={(1, 4): 3.0}, precision=1000.0, k_routes=2
        ).solve()
        assert res.success
        assert float(res.route_flows[0]) == pytest.approx(5.0 / 3.0, abs=1e-3)
        assert float(jnp.sum(res.route_flows)) == pytest.approx(3.0)
        assert res.total_cost == pytest.approx(float(res.route_flows @ res.route_costs))

    def test_parallel_summary_is_stable(self):
        res = sq.RoutingProblem(
            network=PARALLEL_EDGES, demand={(1, 4): 3.0}, precision=100.0, k_routes=2
        ).solve()
        assert str(res.summary()) == PARALLEL_SUMMARY

    def test_edge_list_needs_demand(self):
        with pytest.raises(ValueError, match="needs demand="):
            sq.RoutingProblem(network=PARALLEL_EDGES)

    def test_bad_edges_are_named(self):
        with pytest.raises(ValueError, match="edge 1: capacity must be > 0"):
            sq.RoutingProblem(
                network=[(1, 2, 1.0, 1.0), (2, 3, 1.0, 0.0)],
                demand={(1, 3): 1.0},
            )

    def test_non_convergence_is_a_flag_and_a_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = sq.RoutingProblem(
                network=PARALLEL_EDGES,
                demand={(1, 4): 3.0},
                precision=1000.0,
                k_routes=2,
                max_iter=1,
            ).solve()
        assert res.success is False
        assert any(issubclass(w.category, ConvergenceWarning) for w in caught)


def blotto_payoff(own, rival, values):
    """The Blotto rule, written out independently of the library."""
    return sum(
        value if a > b else (value / 2.0 if a == b else 0.0)
        for a, b, value in zip(own, rival, values, strict=True)
    )


class TestAllocation:
    def test_allocation_matches_the_enumerated_best_response(self):
        values = (1.0, 1.0, 2.0)
        res = sq.AllocationProblem(budget=5, field_values=list(values), precision=2.0).solve()
        assert res.success

        own_grid = [tuple(int(a) for a in row) for row in res.allocations]
        rival_grid = [tuple(int(a) for a in row) for row in res.rival_allocations]
        assert len(own_grid) == math.comb(5 + 2, 2)
        assert all(sum(a) == 5 for a in own_grid)

        mix = [float(p) for p in res.rival_distribution]
        best_value, best_allocation = -math.inf, None
        for own in own_grid:
            value = sum(
                blotto_payoff(own, rival, values) * p
                for rival, p in zip(rival_grid, mix, strict=True)
            )
            if value > best_value:
                best_value, best_allocation = value, own
        assert tuple(int(a) for a in res.allocation) == best_allocation
        assert res.expected_value == pytest.approx(best_value, rel=1e-9)
        assert 0.0 <= res.win_probability <= 1.0

    def test_symmetric_blotto_splits_the_stake(self):
        res = sq.AllocationProblem(budget=4, n_fields=3, precision=1.5).solve()
        own = res.allocation_distribution
        assert float(jnp.max(jnp.abs(own - res.rival_distribution))) < 1e-9
        assert float(own @ res.value_curve) == pytest.approx(1.5, rel=1e-8)

    def test_budget_and_fields_are_validated(self):
        with pytest.raises(ValueError, match="at least 2 fields"):
            sq.AllocationProblem(budget=3, field_values=[1.0])
        with pytest.raises(ValueError, match="budget must be >= 1"):
            sq.AllocationProblem(budget=0, n_fields=3)
        with pytest.raises(ValueError, match="exceeds the dense limit"):
            sq.AllocationProblem(budget=40, n_fields=4, precision=1.0).solve()


class TestElectricity:
    def test_uniform_offers_give_the_minimum_of_two_draws(self):
        """At vanishing precision both units mix uniformly, so the clearing price is
        the expectation of the minimum of two independent uniform grid draws."""
        offers = [20.0, 25.0, 30.0, 35.0, 40.0]
        res = sq.ElectricityProblem(
            costs=[20.0, 20.0],
            offers=offers,
            demand=50.0,
            capacities=[100.0, 100.0],
            precision=1e-9,
        ).solve()
        m = len(offers)
        expected = sum(offer * (2 * (m - k - 1) + 1) / m**2 for k, offer in enumerate(offers))
        assert res.clearing_price == pytest.approx(expected, rel=1e-6)
        assert res.dispatch_probability == pytest.approx(0.5, abs=1e-6)

    def test_symmetric_generators_share_the_dispatch(self):
        """Identical costs must give identical offer ladders, whichever unit you ask about."""
        common = {
            "costs": [20.0, 20.0],
            "offers": (20.0, 60.0, 5.0),
            "demand": 80.0,
            "capacities": [100.0, 100.0],
            "precision": 0.05,
        }
        first = sq.ElectricityProblem(**common, generator=0).solve()
        second = sq.ElectricityProblem(**common, generator=1).solve()
        assert first.success
        ladder = first.offer_curve[:, 1]
        assert float(jnp.max(jnp.abs(ladder - second.offer_curve[:, 1]))) < 1e-9
        assert first.offer == second.offer
        assert first.clearing_price == pytest.approx(second.clearing_price)
        assert float(jnp.sum(ladder)) == pytest.approx(1.0)
        assert float(jnp.sum(first.clearing_price_distribution[:, 1])) == pytest.approx(1.0)
        assert 0.4 < first.dispatch_probability < 0.6
        assert first.revenue > first.profit > 0.0

    def test_demand_above_capacity_is_refused(self):
        with pytest.raises(ValueError, match="exceeds the smaller capacity"):
            sq.ElectricityProblem(
                costs=[20.0, 22.0],
                offers=(20.0, 40.0, 5.0),
                capacities=[10.0, 100.0],
                demand=50.0,
            )

    def test_two_generators_only(self):
        with pytest.raises(ValueError, match="two generators"):
            sq.ElectricityProblem(costs=[20.0, 22.0, 25.0], offers=(20.0, 40.0, 5.0))


class TestConventions:
    def test_every_problem_class_is_re_exported(self):
        for name in (
            "PricingProblem",
            "AuctionProblem",
            "RoutingProblem",
            "AllocationProblem",
            "ElectricityProblem",
            "LogitDemand",
            "LinearDemand",
            "CustomDemand",
            "ConvergenceWarning",
        ):
            assert hasattr(sq, name), name
            assert name in sq.__all__

    def test_summaries_are_short_tables(self):
        solutions = [
            canonical_pricing().solve(),
            sq.AuctionProblem(values=[9.0, 9.0], grid=(4.0, 9.0, 1.0), precision=1.0).solve(),
            sq.AllocationProblem(budget=3, n_fields=3, precision=1.0).solve(),
            sq.ElectricityProblem(
                costs=[20.0, 21.0], offers=(20.0, 40.0, 5.0), precision=0.1
            ).solve(),
            sq.RoutingProblem(
                network=PARALLEL_EDGES, demand={(1, 4): 3.0}, precision=10.0, k_routes=2
            ).solve(),
        ]
        for res in solutions:
            lines = str(res.summary()).splitlines()
            assert 3 <= len(lines) <= 15
            assert lines[1] == "=" * 62
            assert lines[-1] == "=" * 62
            assert repr(res) == str(res.summary())
            assert isinstance(res.as_dict(), dict)
            assert res.as_dict()["success"] is True

    def test_solutions_are_frozen(self):
        res = canonical_pricing().solve()
        with pytest.raises(AttributeError):
            res.price = 2.0
