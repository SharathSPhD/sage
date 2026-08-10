# Estimating dissipation from trajectories

The [exact dissipation meter](dissipation.md) needs the generator — every
transition rate of the joint revision process. Real data gives you none of
that: only an observed sequence of profiles. This page covers the two
estimators that bridge the gap, and exactly how far each can be trusted.
Both are validated against the exact Schnakenberg EPR on synthetic Glauber
trajectories **before** either touches data (PROGRAMME v3 §3.5 discipline).

## The sampling bridge

`strataq.core.dynamics.sample` simulates the Glauber jump process by
uniformisation: the skeleton chain P = I + L/Λ stepped with Exponential(Λ)
holding times — the same law as the CTMC. Two exact facts make the skeleton
the right object for estimation:

- it shares the CTMC's stationary distribution;
- its per-step entropy production is exactly EPR/Λ, and for a stationary
  Markov chain the (k+1)-block KLD equals k × (per-step EP) — so the KLD
  estimator has a sharp target at **every** k, not an asymptote.

## KLD / k-th order Markov (`kld_epr`)

Plug-in estimate of σ̂⁽ᵏ⁾ = (1/kτ) Σ P(Y₀:ₖ) log P(Y₀:ₖ)/P(Yₖ:₀).
Assumption-light and data-hungry: bias is O(n_cells / n_samples).

Validation reading (`estimator_alpha_sweep.json`): Spearman ρ(KLD, exact) =
1.0 across ten α levels, with per-level agreement to ~1% at 8×60k steps.
On exact potential games it reads < 5×10⁻³ (pure plug-in bias; the true
value is 0). Its weakness is the other direction — under partial
observation (hidden states, coarse-grained prices) it underestimates, which
is why it is not the headline number for data.

## TUR lower bound (`tur_epr_bound`)

σ ≥ 2⟨J_T⟩²/(Var(J_T)·T) from the first two cumulants of an empirical
time-integrated current. A **certified bound**, not a point estimate: a bad
current choice loosens it but never breaks it, which is why it is the
headline number for partial-observation data.

Three honesty rules learned the hard way (each is a test):

1. **Fix the horizon, not the jump count.** The TUR is a fixed-T statement;
   fixing the number of jumps instead suppresses the Poisson event-count
   fluctuation, underestimates Var(J), and pushes the "bound" *above* the
   true EPR (observed: ×1.5 overshoot before the fix). `window_currents`
   truncates all windows at a common T.
2. **Debias the cumulants.** E[J̄²] carries a +Var/M excess and 1/V̂ar a
   Jensen factor (M−1)/(M−3); at M ≲ 32 windows these alone push a
   saturated bound past the truth.
3. **Certify through a CI, not the point value.** Near equilibrium
   (α → 0) the TUR saturates — the true ratio bound/EPR approaches 1 — so
   the `tur_epr_bound` point estimate straddles 1 legitimately (observed
   above exact at 2 of 10 sweep levels, max ratio 1.07 even after
   debiasing). The certifiable statement is the lower bootstrap quantile
   from `tur_epr_bound_ci`; the gate criterion is that quantile ≤ exact
   EPR at every level.

Tightness is itself a diagnostic: ~0.97 at α = 0.05 (linear-response
saturation), drifting to ~0.6–0.7 by α = 0.95. The bound degrades exactly
where dissipation is largest — gracefully, and in the direction that keeps
it conservative.

## Weight choice

`stationary_current_weights` (sign of the exact J*) is **oracle-informed**
— for validation and synthetic tightness studies only.
`empirical_flux_weights` derives weights from the data's own net flux; for
strict certification derive them on a held-out split. The NEEP-style neural
estimator from the programme is deferred (an ADR is required to start it).
