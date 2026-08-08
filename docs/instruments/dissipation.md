# The dissipation meter — currents and entropy production

**What it reads.** Whether the strategic dynamics are in thermodynamic equilibrium or merely stationary. The Glauber-logit chain (a random player revises to a logit choice over pure payoffs) has a stationary distribution π over joint profiles; the meters are the **probability current** $J^*(a,a') = \pi(a)w(a{\to}a') - \pi(a')w(a'{\to}a)$ and the **entropy production rate**

$$\sigma_{\mathrm{EP}} = \tfrac12 \sum_{a,a'} \big[\pi(a)w(a{\to}a') - \pi(a')w(a'{\to}a)\big]\,\log\frac{\pi(a)w(a{\to}a')}{\pi(a')w(a'{\to}a)} \;\ge\; 0 .$$

**Why it means something** *(tier: exact — K3 + Schnakenberg network theory)*. In an exact potential game the logit revision probabilities are the heat-bath conditionals of $\pi \propto e^{\lambda\Phi}$: the chain is reversible, $J^* = 0$, $\sigma_{\mathrm{EP}} = 0$ — strategic equilibrium *is* thermodynamic equilibrium. Off potentiality, detailed balance breaks: the system holds a non-equilibrium steady state that continuously circulates (Edgeworth-style cycling lives here) and dissipates.

**Calibration state.** Gate `dynamics.exact`:

| Reading | Requirement | Artifact |
|---|---|---|
| π vs e^{λΦ}/Z, congestion n=2/3, λ ∈ {0.7, 1.2, 2} | ≤ 10⁻¹⁰ | `gibbs_agreement.json` |
| Potential games | EPR and \|J*\| < 10⁻¹² | `equilibrium_reads_zero.json` |
| RPS, matching pennies | EPR > 10⁻³, circulation present | `ness_reads_positive.json` |
| 1,000-game α sweep at fixed λ = 1.2 | ρ(EPR, α) > 0.9 **and** ρ(EPR, ℛ) > 0.8 | `chain_comovement.json` |

Measured, honestly split: marginally ρ(EPR, α) = 0.990 and ρ(EPR, ℛ) = 0.993 — but the second number is α-driven. **Stratified by α, within-level ρ(EPR, ℛ) is +0.80..+0.88 up to α ≈ 0.65, degrades above, and reverses to −0.355 at α = 0.95** (per-level values in the artifact): among near-pure-harmonic games the reciprocity and dissipation meters *decouple* — the first realisation of C1's falsifier, and the programme's first genuine finding (F-0004). Working hypothesis under chase: at α → 1, ℛ's symmetric denominator underflows, so ℛ reads the smallness of the residual potential part rather than circulation strength. The two meters are complementary, not substitutes.

Artifacts regenerate via `uv run python -m experiments.dynamics_calibration` (`make reproduce`).

**Limitations, stated once.** This is the *exact, generator-level* meter: it needs the full profile space, so it runs on small games. Real-data dissipation goes through the trajectory estimators (KLD, TUR bound, NEEP — a later unit) which are validated against this meter on synthetic ground truth before touching anything empirical. The co-movement result is rank-order evidence at fixed λ, on 3×3 two-player families.
