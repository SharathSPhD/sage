# API reference — the public surfaces

Two public contracts, kept deliberately small. Anything not listed here is
library-internal and may move; these signatures are stable within 0.x.

## `strataq.toolkit` (Python)

Plain lists/numpy in, frozen dataclasses out. All entry points validate
loudly (NaN/inf, shapes, constant series → instructive `ValueError`) and
attach honesty warnings to results.

### `estimate_rationality(payoff_matrices, counts, *, lam_range=(0.05, 20.0), grid_points=400) → RationalityEstimate`

Bayesian posterior over the logit rationality λ from per-player choice
tallies under a known game. Fields: `mean`, `map`, `ci_low`, `ci_high`
(calibrated 95% credible interval — coverage 48/46/48 per λ* in the
`bayes_recovery` artifact), `grid_resolved`, `warnings`. Guards: the scale
fold (λ is per payoff unit — always warned), flat likelihood (warned, never
a bare number), grid resolution (auto-refined; flagged if still limited).

### `reciprocity_read(chi, *, chi_se=None, n_draws=2000, seed=0) → ReciprocityRead`

ℛ = ‖χ − χᵀ‖/‖χ + χᵀ‖ from any square cross-response matrix. With
`chi_se` (elementwise standard errors): Monte-Carlo 95% CI and an
uncertainty-aware verdict that refuses to classify across bands. Without:
a self-declared point read; near-threshold values labelled borderline.
Fields: `r`, `verdict`, `ci_low`, `ci_high`, `calibration` (the committed
bracket: road network 0 / Blotto 0.12 / RPS 0.69), `warnings`.

### `irreversibility_test(series, *, n_bins=3, n_surrogates=200, alpha_level=0.01, seed=0) → ReversibilizedNullResult`

Phase-embedded KLD irreversibility vs the reversibilized-Markov null
(detailed-balance-exact, persistence-matched — the F-0009 instrument).
Fields: `detected`, `p_value`, `statistic`, `null_quantile`, `null_median`,
`null_mismatch_low`, `n_surrogates`. Power: ≥ 80% at n ≥ 300 (measured);
constant or NaN series raise.

### `game_thermo(payoff_matrices, lam=1.5) → GameThermoRead`

One-call dashboard: `alpha` (harmonic fraction), `r` (reciprocity defect at
the QRE), `epr` (entropy production of the joint revision dynamics),
`verdict`.

## `strataq.thermo.hs_estimator` (did my data settle?)

### `relaxation_gate(windows, *, n_states, hold_durations, relax_safety=4.0, se_method="split", se_sigma=2.0) → RelaxationGate`

The settling check, usable on its own. Any plug-in estimate of a
*stationary* quantity — an occupation distribution, a stationary current, a
Hatano–Sasa Y — is meaningless if the system never reached stationarity
inside the observation window, and this answers that question from the state
sequences alone. Returns `ok`, plus per-window `tau_hats`, `ses`,
`thresholds` (`tau_hat + se_sigma x SE`, what the gate actually compares
against the hold) and the `offenders` list, so a refusal tells you *which*
hold was too short and by how much.

`se_method` selects the error bar on the relaxation-time estimate:

| method | order-invariant? | notes |
|---|---|---|
| `split` | **no** | the incumbent 4-way `i::4` trajectory split; the default, so no previously recorded verdict moves |
| `jackknife` | yes, exactly | leave-one-out over trajectories in closed form; also carries the pi-hat noise |
| `delta` | yes, exactly | cheapest, but treats pi-hat as fixed — measured to *overstate* the SE, since a trajectory's occupancy moves the match rate and the baseline together |
| `bootstrap` | in distribution only | trajectory resampling; a fixed seed leaves an O(SE/sqrt(2B)) residual — measured at 0/20 flips anyway. **RECOMMENDED** |

**Use `bootstrap`. The default is `split` only so previously published reads
reproduce.** On fast-mixing windows the lag-N/4 autocorrelation has already
decayed into noise, so ρ sits at or below zero and τ̂ returns a clip-floor
artifact instead of a relaxation time (F-0021). Every SE that depends on local
sensitivity to ρ fails there for the same underlying reason: `delta` explodes
(its gradient divides by ρ), `jackknife` returns an SE of **exactly zero** on
6 of 20 seeds at n=30 — every leave-one-out replicate pins to the same floor,
so the gate is told τ̂ is known perfectly and drops its noise margin — and
`split` does so on 2 of 20. `bootstrap` never collapses (0 of 20) and is also
the most accurate against an independently measured oracle SE (0.18–0.29
relative deviation, against `split`'s 0.44–0.49). A consequence worth knowing
when you read a verdict: because τ̂ degenerates on the fast windows, the gate
is in practice testing only the slow ones, and it does not currently say so.

Order-invariance matters because the trajectory ORDER is not a physical
property: permuting it must not change a verdict, and with `split` it can
(R8/F-0019 measured 6 flips in 20 seeds at n=30). See F-0019/F-0020 and
`config/experiments/gate_se.yaml` for the registered evidence.

## `strataq.estimate.bayes` (power users)

`grid_posterior` / `refined_posterior` (the resolution-guard-enforcing
entry point), `log_evidence`, `log_evidence_mixture`, `bayes_factor`,
`precompute_sigmas`, and the EFE campaign loop: `Hypothesis`,
`efe_scores`, `update_beliefs`,
`run_campaign(hypotheses, probes, *, run_probe, sigma, budget,
stop_confidence=0.95, min_probes=1, prior=None)` — `min_probes` is a
stopping gate (the F-0014 lesson); held-out validation of the winner is
the caller's job. Campaign results carry the full audit trail.

## HTTP API (`/v1/toolkit`, live)

Base: `https://sage-labs.vercel.app/api` (proxies the float64 backend).
Validation errors return 422 with the same instructive messages; every
response carries `warnings`.

| Endpoint | Body | Returns |
|---|---|---|
| `POST /v1/toolkit/reciprocity` | `{chi, chi_se?}` | `r, verdict, ci_low, ci_high, calibration, warnings` |
| `POST /v1/toolkit/irreversibility` | `{series, n_bins?, n_surrogates?, alpha_level?, seed?}` | `detected, p_value, statistic, null_*, warnings` |
| `POST /v1/toolkit/rationality` | `{payoff_matrices, counts, lam_min?, lam_max?}` | `mean, map, ci_low, ci_high, grid_resolved, warnings` |
| `POST /v1/domains/blotto/read` | `{budget_a, budget_b, n_fields?, field_values?, lam?}` | allocation mixes, `alpha, r, epr?, warnings` |

Older instrument endpoints (`/v1/solve/qre`, `/v1/decompose`,
`/v1/response`, `/v1/response/poke`, `/v1/dynamics/*`,
`/v1/estimate/lambda`, `/v1/domains/sioux_falls/*`) are documented in the
service's OpenAPI (`/docs` on the backend host). Size guards apply
(≤ 3 players, ≤ 12 actions/player, dense dynamics ≤ 400 joint states).
