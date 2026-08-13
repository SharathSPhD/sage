# Solving problems

Five problem types, one shape: build a `Problem`, call `.solve()`, read a frozen
`Solution` whose fields are named for your domain. `.summary()` prints a compact
table. Bad input raises `ValueError`; a solve that misses tolerance sets
`success = False` and emits a `ConvergenceWarning`.

```python
import strataq as sq
```

## Pricing

Set a price against rivals who are also setting prices.

```python
prob = sq.PricingProblem(
    costs=[1.00, 1.05],
    grid=(1.09, 1.89, 0.10),
    demand=sq.LogitDemand(price_sensitivity=3.6, quality=[0.0, -0.1]),
    precision=1.5,
)
res = prob.solve()

res.price  # 1.29 — the price to set
res.profit  # expected profit at that price
res.margin  # price minus your marginal cost
res.rival_prices  # (n_rivals, n_levels) distribution over each rival's price
res.expected_rival_prices  # each rival's mean price
res.profit_curve  # profit at every level of the grid
res.elasticities  # own on the diagonal, cross off it
print(res.summary())
```

```text
                    strataq PricingProblem
==============================================================
price                        1.29   firms                    2
profit                   0.002746   price levels             9
margin                       0.29   precision              1.5
own elasticity               -4.6   demand          LogitDemand
cross elasticity          0.02242   mean rival price      1.49
==============================================================
```

Demand comes from `LogitDemand`, `LinearDemand`, or your own function:

```python
import jax.numpy as jnp

sq.LinearDemand(intercept=[10.0, 10.0], own_slope=1.0, cross_slope=0.4)
sq.CustomDemand(lambda p: jnp.maximum(10.0 - p + 0.4 * (jnp.sum(p) - p), 0.0))
```

With one firm the problem is a monopoly and `res.price` is the profit-maximising
level on the grid — `costs=[2.0]`, `LinearDemand([10.0], 1.0)` returns `6.0`.

`grid` is `(start, stop, step)` as a tuple, or an explicit list of levels.

## Auctions

Sealed-bid sale (`values=`) or procurement tender (`costs=`).

```python
res = sq.AuctionProblem(
    values=[10.0, 10.0, 10.0],  # or values=10.0 with n_bidders=3
    grid=(5.0, 10.0, 0.5),
    reserve=6.0,
    precision=2.0,
).solve()

res.bid  # the bid to submit
res.surplus  # expected surplus
res.win_probability
res.rival_bids  # (n_rivals, n_levels)
res.expected_clearing_bid
```

Procurement flips the sign and the direction: `costs=[3.0, 3.2]` with
`reserve=9.0` treats the reserve as the buyer's ceiling, and the lowest eligible
offer wins.

## Routing

Traffic assignment with optional tolls, on TNTP data or your own edge list.

```python
res = sq.RoutingProblem(network="sioux_falls", precision=0.5, max_od=12).solve()

res.flows  # link flows
res.travel_times  # link travel times at those flows
res.total_cost  # total travel time
res.mean_travel_time
print(res.summary())
```

```text
                    strataq RoutingProblem
==============================================================
total cost              2.382e+05   links                   76
mean travel time             5.45   routes                  36
max volume/capacity         1.241   od pairs                12
toll revenue                    0   precision              0.5
cost change                    --   converged             True
==============================================================
```

Toll a link and read what it did:

```python
tolled = sq.RoutingProblem(network="sioux_falls", tolls={28: 5.0}).solve()
tolled.toll_effect.revenue  # toll x flow on the tolled links
tolled.toll_effect.delta_total_cost  # change in total travel time
tolled.toll_effect.delta_flows  # per-link change
```

A plain edge list works the same way — `(from, to, free_flow, capacity)` or
`(from, to, free_flow, capacity, b, power)`, or dicts with those keys:

```python
res = sq.RoutingProblem(
    network=[
        (1, 2, 1.0, 1.0, 1.0, 1.0),
        (1, 3, 2.0, 1.0, 0.25, 1.0),
        (2, 4, 0.0, 1.0, 0.0, 1.0),
        (3, 4, 0.0, 1.0, 0.0, 1.0),
    ],
    demand={(1, 4): 3.0},
    precision=100.0,
    k_routes=2,
).solve()
res.route_flows  # [1.665, 1.335] — the analytic equilibrium is 5/3, 4/3
```

Route sets are the `k_routes` shortest paths by free-flow time per OD pair.

## Allocation

Colonel Blotto: split a budget across contested fields.

```python
res = sq.AllocationProblem(budget=5, field_values=[1.0, 1.0, 2.0], precision=2.0).solve()

res.allocation  # e.g. [0, 1, 4]
res.win_probability  # of capturing more than half the total value
res.expected_value
res.rival_distribution  # the mix you are playing against
```

## Electricity

Offer a block into a uniform-price market.

```python
res = sq.ElectricityProblem(
    costs=[20.0, 22.0],
    offers=(20.0, 60.0, 5.0),
    capacities=[100.0, 100.0],
    demand=80.0,
    precision=0.05,
).solve()

res.offer_curve  # (n_offers, 2) of [price, probability]
res.clearing_price  # expected clearing price
res.clearing_price_distribution  # (n_offers, 2) of [price, probability]
res.revenue  # expected revenue
res.dispatch_probability
```

## The physics, if you want it

Every solution carries a `.diagnostics` object, computed lazily on first access
and never on the solve path. Nothing there affects the answer.

```python
res.diagnostics.alpha  # harmonic fraction of the game
res.diagnostics.reciprocity_defect  # response asymmetry
res.diagnostics.entropy_production  # dissipation of the Glauber dynamics
res.diagnostics.rho_sb  # spectral radius, distance to criticality
res.diagnostics.residual, res.diagnostics.n_iter
```

## Failure

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    res = sq.PricingProblem(
        costs=[1.00, 1.05],
        grid=(1.09, 1.89, 0.10),
        demand=sq.LogitDemand(price_sensitivity=3.6, quality=[0.0, -0.1]),
        precision=1.5,
        max_iter=1,  # stop the solver before it gets there
    ).solve()

res.success  # False
res.message  # "PricingProblem.solve did not converge: residual 5.6e-02 exceeds ..."
caught[0].category  # strataq.ConvergenceWarning
```

Bad input never reaches the solver:

```python
sq.PricingProblem(costs=[1.0], grid=(2.0, 1.0, 0.5), demand=...)
# ValueError: grid stop must be >= start, got (2.0, 1.0)
```

## Over HTTP

Each problem type has an endpoint that mirrors the constructor and returns the
solution as JSON. Grids arrive as explicit levels (`grid`) or as a range
(`grid_range`), never both.

```sh
curl -X POST localhost:8000/v1/solve/pricing -H 'Content-Type: application/json' -d '{
  "costs": [1.00, 1.05],
  "grid_range": [1.09, 1.89, 0.10],
  "demand": {"kind": "logit", "price_sensitivity": 3.6, "quality": [0.0, -0.1]},
  "precision": 1.5
}'
```

`POST /v1/solve/pricing`, `/v1/solve/auction`, `/v1/solve/routing`,
`/v1/solve/allocation`, `/v1/solve/electricity`. Add `"diagnostics": true` to
get the `.diagnostics` block in the response.
