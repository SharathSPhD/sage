# Cookbook — the instruments on your data

Five-minute recipes. Everything here uses `strataq.toolkit` (plain lists in,
verdicts out) or the public HTTP API; the numbers come from the same gated,
red-teamed machinery as the research findings, and every result carries its
honesty warnings. Install:

```bash
pip install strataq
```

No Python? Every recipe has a `curl` twin against the live API
(`https://sage-labs.vercel.app/api` proxies it), and a point-and-click twin
at [sage-labs.vercel.app/tools](https://sage-labs.vercel.app/tools).

## Is my market / ecosystem a landscape or a whirlpool?

You have a measured **cross-response matrix** — how each agent's action moves
when another agent's incentives shift. Cost pass-through between firms is the
canonical example; any small-shift response estimate works.

```python
import strataq.toolkit as tk

# the actual Dominick's grocery estimates (finding F-0011):
read = tk.reciprocity_read(
    [[1.0697, 0.0028], [0.0005, 0.9685]],
    chi_se=[[0.02, 0.001], [0.001, 0.02]],  # your regression's standard errors
)
print(read.r)  # 0.0011
print(read.verdict)  # reciprocal (landscape-like) — 95% CI [...]
print(read.warnings)  # the fine print travels with the number
```

```bash
curl -X POST https://sage-labs.vercel.app/api/v1/toolkit/reciprocity \
  -H 'Content-Type: application/json' \
  -d '{"chi": [[1.0697, 0.0028], [0.0005, 0.9685]]}'
```

!!! warning "Read the warnings"
    Without `chi_se` the verdict is a point read of a noisy estimate and says
    so. Only the *zero test* of ℛ is λ-free; compare magnitudes only at
    matched conditions.

## Is my time series irreversibly driven?

Any scalar series ordered in time — prices, order flow, load, sentiment.
The test phase-embeds it (value bins alone are provably blind to loop
irreversibility) and compares against reversible surrogates with matched
persistence. This is the instrument that found the day-ahead electricity
market's diurnal loop (F-0009) and certified retail category prices at-null
(F-0011's companion scan).

```python
verdict = tk.irreversibility_test(weekly_prices, n_bins=3, n_surrogates=200)
verdict.detected  # True: no reversible chain with these pair statistics does this
verdict.p_value
```

Power: ≥ 80% detection at n ≥ 300 on a known driven series; n ≈ 100 is
underpowered and the API will tell you so. NaNs and constant series raise —
the instruments never compute through bad input silently.

## How payoff-sensitive are my agents?

You know the game's payoffs and observed choice frequencies; you want the
logit rationality λ **with a defensible interval**:

```python
est = tk.estimate_rationality(
    [u_row_player, u_col_player],  # one matrix per player
    counts=[[412, 95, 493], [301, 402, 297]],
)
est.mean, (est.ci_low, est.ci_high)  # calibrated 95% credible interval
est.warnings
```

Two honesty guards you will meet: the **scale fold** (λ is per payoff unit —
rescaling payoffs by *s* rescales λ by 1/*s*; only the product is ever
identified) and the **flat-likelihood warning** (on symmetric games the QRE
can be uniform at every λ; the toolkit warns instead of quoting noise).

## Dashboard for a game you can write down

```python
tk.game_thermo([u1, u2], lam=1.5)
# -> alpha (harmonic fraction), R (reciprocity defect), epr (dissipation), verdict
```

Rock–paper–scissors reads whirlpool (α ≈ 1, positive dissipation); a
coordination game reads landscape (α ≈ 0, zero dissipation to machine
precision); real systems land in between — the calibration bracket is
road-network 0 / Blotto 0.12 / RPS 0.69.

## Going deeper

The facade is a thin layer: `strataq.estimate.bayes` exposes the posterior
machinery and the EFE experiment-selection loop (`run_campaign`),
`strataq.thermo` the exact dissipation meters, quench protocols and
surrogate nulls, `strataq.finite` the solvers and the Hodge decomposition.
Every unit's gate, artifacts and adversarial-review record are in the
[progress dashboard](progress/index.md).
