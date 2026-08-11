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
