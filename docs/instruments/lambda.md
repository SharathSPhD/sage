# Estimating λ — four routes and an honesty protocol

λ is the most abused dial in the QRE literature: mis-specified demand,
unmodelled heterogeneity and wrong payoff models all try to leak into it.
The defence (`strataq.estimate`) is redundancy — four structurally different
estimators, run together, with disagreement treated as a *diagnostic*:

| Route | Uses | Assumes | Fails how |
|---|---|---|---|
| `lambda_mle` | choice frequencies | payoffs known, single-λ QRE | flat likelihood ⇒ **warns** "unidentified" |
| `lambda_mle_implicit` | same, scored by autodiff through an unrolled solve | same | must agree with the grid — a solver cross-check |
| `lambda_moment_chi` | an observed cross-response matrix (pass-through) | responses measurable | needs perturbation data |
| `lambda_dispersion` | mean choice entropy only | entropy varies with λ | flat entropy curve ⇒ **warns** (e.g. a symmetric principal branch below its pitchfork) |

**Validation** (`estimator_recovery.json`, seed 20260810, 20 cases over an
asymmetric anchor game plus α ∈ {0.15, 0.45, 0.75} family games, λ\* from
0.3 to 6): median relative error 2.6% (MLE), 2.8% (dispersion), exact for
χ-matching from oracle responses.

**The diagnostic fires when it must** (`estimator_misspecification.json`):
on data drawn from a λ-*mixture* (half 0.4, half 4.0 — not a QRE at any
single λ) the estimator spread widens ×91 over clean data, and the protocol
flags it instead of averaging. On symmetric RPS — where every λ produces the
uniform mix and λ is simply not identified from frequencies — every route
returns a **warning, not a number**.

Two rules for data use (Stage 3): report the whole family, never one number;
and treat a flagged disagreement as a finding about the model, not a nuisance
to be smoothed.

## The Bayesian layer and the EFE experiment chooser (unit estimate.bayes, ADR-0012)

Point estimators became posteriors: `grid_posterior` puts an exact discrete posterior over λ on a log grid (uniform-on-grid prior, deliberately — it makes the F-0006 scale fold an *exact reparameterisation in posterior space*: the posterior under payoffs s·u on grid g equals the posterior under u on grid s·g, weight for weight, tested to 1e-8). The posterior self-diagnoses grid coarseness: fewer than ~2 effective grid points (participation ratio) flags `grid_resolved = False`, and the credible interval must not be quoted until the grid is refined — coverage is calibrated (≥ 8/10 seeds per λ*, artifact `bayes_recovery.json`) only under that guard. R1's mixture-misspecification diagnostic is now a matched model comparison: an explicit two-λ mixture model decisively beats single-λ on mixture data (BF > 100) and is Occam-suppressed on clean data (BF < 10).

The same unit carries the **EFE experiment chooser** (the active-inference pattern: competing quantitative hypotheses each predict every candidate probe's outcome; the next probe maximises BALD mutual information; beliefs update by Bayes; the campaign stops on concentration or budget). Its first campaign adjudicates F-0012's open mechanism among four pre-declared hypotheses — potential-scale fold, NESS-sensitivity floor, spectral gap, quadratic strawman — with a pre-registered absolute adequacy guard (a winner that doesn't actually fit is recorded as "all hypotheses inadequate") and a σ-sensitivity re-run. Artifact: `efe_mechanism_campaign.json`, full audit trail per round.
