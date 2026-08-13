# Solving problems

Nine problem types, one shape: build a `Problem`, call `.solve()`, read a frozen
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

## Repeated games

Will the arrangement hold? A stage game plus a discount factor, and the two
questions the folk theorem asks: is this profile self-enforcing at your discount
factor, and how patient would the parties have to be.

```python
prisoners_dilemma = [
    [[3.0, 0.0], [5.0, 1.0]],  # you: cooperate, defect
    [[3.0, 5.0], [0.0, 1.0]],
]
res = sq.RepeatedProblem(payoffs=prisoners_dilemma, discount=0.6).solve()

res.critical_discount  # 0.5 — the folk-theorem answer for these payoffs
res.sustainable  # True: 0.6 >= 0.5
res.critical_by_player  # who binds, if the players are not symmetric
res.deviation_payoffs  # what a one-shot deviation is worth to each
res.punishment_payoffs  # the minmax they fall back to
res.sustainable_profiles  # every pure profile grim trigger holds up at 0.6
res.frontier  # the Pareto-undominated ones
print(res.summary())
```

```text
                   strataq RepeatedProblem
==============================================================
sustainable                  True   players                  2
critical delta                0.5   profiles                 4
delta                         0.6   sustainable              2
binding player                  0   frontier                 1
cooperation prob               --   precision               --
==============================================================
```

The critical δ is `(d - u) / (d - p)`: deviation payoff, agreed payoff,
punishment payoff. Punishment is the pure-strategy minmax by default;
`punishment="nash"` reverts to the worst pure Nash equilibrium of the stage game
instead, and an explicit vector overrides both.

Add `precision=` and sustainability stops being a bit and becomes a probability —
the logit analogue, where each party chooses between keeping to the arrangement
and deviating with a logit rule over continuation values:

```python
res = sq.RepeatedProblem(payoffs=prisoners_dilemma, discount=0.8, precision=5.0).solve()
res.cooperation_probability  # P(the agreed profile is actually played)
```

Strategies other than grim trigger are finite automata, and the same incentive
machinery applies to them:

```python
from strataq.repeated import critical_discount, is_sustainable, tit_for_tat

game = sq.DenseTensorGame(prisoners_dilemma)
tft = tuple(tit_for_tat(game.num_actions, i) for i in range(2))
is_sustainable(game, tft, 0.9)  # False — tit-for-tat is not subgame perfect
```

That is the correct answer, not a bug: after a lone defection, tit-for-tat puts
the players in an alternating punishment worth less than simply forgiving, so the
one-shot deviation criterion fails at that history however patient they are.

### Edgeworth price cycles

Prices in retail fuel, airline seats and electricity do not settle; they saw
upward and grind down. Run alternating logit best response on a price ladder and
measure what comes out:

```python
from strataq.repeated import edgeworth_cycle, linear_market_demand

cycle = edgeworth_cycle(
    costs=[1.0, 1.0],
    ladder=[1.0 + 0.2 * i for i in range(11)],
    demand=linear_market_demand(intercept=10.0, slope=1.0),
    lam=60.0,
    capacities=[5.0, 5.0],
)

cycle.period  # length of the cycle in revision steps
cycle.amplitude  # peak minus trough of the market price
cycle.mean_price, cycle.peak, cycle.trough
cycle.price_path  # (steps, firms) — the whole trajectory
cycle.is_fixed_point  # False when it really cycles
```

Capacity is what makes the cycle. With unlimited capacity the undercutting stops
one tick above cost, because matching and splitting beats undercutting to a zero
margin — textbook Bertrand, no cycle. Give each firm less capacity than the
market and the expensive firm still sells the residual, so somebody always has a
reason to relent and jump back to the top. λ is the only other knob: large λ is
the sharp sawtooth, small λ smooths it into a fixed point.

## Evolutionary dynamics

What survives copying, rather than what rational players would choose.

```python
stag_hunt = [[3.0, 0.0], [2.0, 2.0]]  # stag, hare — hare is risk dominant
res = sq.EvolutionaryProblem(payoff=stag_hunt, intensity=1.0, population=50).solve()

res.rest_points  # every replicator rest point, found by support enumeration
res.kinds  # 'stable' / 'unstable' / 'saddle' / 'centre' / 'degenerate'
res.stable  # the asymptotically stable ones
res.fixation_a  # probability one stag-player takes over a hare population
res.fixation_b
res.moran_stationary  # the whole distribution over the count of stag-players
res.moran_share  # E[i/N]
```

Rest points are exact: on each support the condition `(Ax)_i = x·Ax` is a small
linear solve, so the answer is the complete list rather than wherever an
integrator happened to stop. Stability is the spectrum of the replicator
Jacobian on the simplex tangent space — at a vertex those eigenvalues are
literally the invasion fitnesses `A[i,k] - A[k,k]`.

```python
from strataq.evolutionary import replicator_flow, rest_points, stability

rock_paper_scissors = [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]]
points = rest_points(rock_paper_scissors)  # three vertices plus the centre
[p.kind for p in points]  # the vertices are saddles, the centre is a centre
replicator_flow(rock_paper_scissors, [0.5, 0.3, 0.2], step=0.001, steps=4000)
```

### The selection intensity is the logit precision

The Fermi rule — a player copies a better-performing neighbour with probability
`1 / (1 + e^{-β Δπ})` — is *identically* the two-action logit choice probability
at λ = β. Not an analogy: the same function. So a game has an evolutionary
reading and a strategic-form reading at the same number, and the library reports
both with the gap between them:

```python
res.fermi_gap  # |imitation probability - logit choice probability|, ~1e-16
res.qre_gap  # |logit-dynamic rest point - symmetric logit QRE at lambda = beta|
res.logit_rest_point  # x solving x = softmax(beta A x)
res.qre_symmetric  # the same fixed point reached through strataq.finite
```

Both are zero to solver tolerance, and the tests assert it. The bridge runs
through the *logit* dynamic, `ẋ = softmax(λ A x) − x`, whose rest points are the
logit QRE; the replicator's own rest points are Nash points and are a different
object, which is why both are exposed rather than conflated.

## Game trees

Order matters and someone cannot see everything: entry then response, offer then
counter-offer, a hand of cards. The answer is a behaviour strategy — what to do
at each information set.

```python
res = sq.ExtensiveProblem(tree="entry_deterrence", precision=2.0).solve()

res.recommended  # most likely action at each information set, by name
res.behaviour  # (n_infosets, max_actions) agent QRE
res.subgame_perfect_actions  # ('in', 'accommodate') — backward induction
res.expected_payoffs
res.divergence  # how far quantal play sits from the textbook answer
print(res.summary())
```

The catalogue is `entry_deterrence`, `centipede`, `bargaining`, `seltens_horse`
and `kuhn_poker`; `options=` passes constructor arguments (`{"n_moves": 6}`).

`precision` solves the **agent QRE** of McKelvey–Palfrey (1998): each information
set is its own decision maker, choosing with a logit rule over the payoff
conditional on reaching it. That is the model that explains the centipede, where
backward induction says stop at the first node and nobody does:

```python
res = sq.ExtensiveProblem(tree="centipede", precision=1.0, options={"n_moves": 6}).solve()
res.subgame_perfect_actions[0]  # 'take'
float(res.behaviour[0, 1])  # P(pass) — well above a half
```

Trees are nested dicts, which is also the JSON the API accepts:

```python
from strataq.extensive import ExtensiveGame, agent_qre, backward_induction

tree = ExtensiveGame.from_dict(
    {
        "players": ["Entrant", "Incumbent"],
        "root": {
            "player": "Entrant",
            "infoset": "enter?",
            "actions": ["in", "out"],
            "children": [
                {
                    "player": "Incumbent",
                    "infoset": "respond",
                    "actions": ["fight", "accommodate"],
                    "children": [{"payoffs": [-1.0, -1.0]}, {"payoffs": [1.0, 1.0]}],
                },
                {"payoffs": [0.0, 2.0]},
            ],
        },
    }
)
tree.to_dict()  # round-trips
backward_induction(tree).value  # [1.0, 1.0]
agent_qre(tree, 2.0).behaviour
```

A chance node is `{"player": "chance", "probs": [...], "children": [...]}`. Nodes
sharing an `infoset` key are indistinguishable to the player who moves there;
that is how `seltens_horse` and `kuhn_poker` are built. Behaviour and mixed
strategies interconvert under Kuhn's theorem
(`behaviour_to_mixed`, `mixed_to_behaviour`, `realisation_gap`), and
`reduced_normal_form` gives the strategic form over pure plans when the tree is
small enough to have one.

Backward induction refuses on trees with non-singleton information sets rather
than pretending; the λ → ∞ limit of the agent QRE is what stands in for it there.

## One call for a whole recommendation

Everything a caller needs to show a user, assembled once so no client assembles
it twice:

```python
res = sq.solve_situation(
    [
        [[2.0, 2.0], [5.0, 0.0]],  # you: hold, gamble
        [[0.0, 1.0], [0.0, 1.0]],  # them: soft, hard
    ],
    actions=["hold", "gamble"],
    rival_actions=[["soft", "hard"]],
    players=["you", "them"],
    precision=1.0,
)

res.action_label  # what to do
res.expected_payoff
res.confidence  # gap to the runner-up, as a fraction of the payoff range
res.alternatives  # every action of yours, best first, each with its regret
res.rivals  # each other party: distribution, most likely action, entropy
res.sensitivity  # does the answer survive a different guess about precision?
print(res.summary())
```

`sensitivity` re-solves across a ladder of precisions around the one you stated
and reports `robustness` (the fraction of the ladder that agrees),
`switch_precision` (where the answer first changes) and `stable`. A
recommendation that holds from λ/4 to 4λ is a different object from one that
holds only at the λ you happened to pick, and the caller should be able to see
which it has.

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
`/v1/solve/allocation`, `/v1/solve/electricity`, `/v1/solve/repeated`,
`/v1/solve/evolutionary`, `/v1/solve/extensive`, `/v1/solve/situation`, plus
`POST /v1/dynamics/edgeworth` and `GET /v1/extensive/catalogue`. Add
`"diagnostics": true` to get the `.diagnostics` block where the problem type
defines one.

```sh
curl -X POST localhost:8000/v1/solve/repeated -H 'Content-Type: application/json' -d '{
  "payoffs": [[[3, 0], [5, 1]], [[3, 5], [0, 1]]],
  "discount": 0.6
}'

curl -X POST localhost:8000/v1/solve/situation -H 'Content-Type: application/json' -d '{
  "payoffs": [[[2, 2], [5, 0]], [[0, 1], [0, 1]]],
  "actions": ["hold", "gamble"],
  "precision": 1.0
}'

curl -X POST localhost:8000/v1/solve/extensive -H 'Content-Type: application/json' -d '{
  "tree": "centipede", "precision": 1.0, "options": {"n_moves": 6}
}'
```
