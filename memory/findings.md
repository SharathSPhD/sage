# Findings — the anomaly log

Anomalies are the product. Every unexpected meter reading lands here with the config that produced it and whether it was chased. Negative results are written plainly, not dressed up.

Entry format:

```
## F-NNNN — <one-line description>
- Date:
- Instrument / experiment:
- Config: <path to resolved config or gate artifact>
- Expected / observed:
- Chase status: chasing | parked (why) | resolved (what it was)
- Resolution / follow-up:
```

## F-0001 — ℛ exceeds 1 on matching pennies (ℛ is not a fraction)

- Date: 2026-08-08
- Instrument / experiment: `reciprocity_defect` via `experiments/reciprocity_calibration.py`
- Config: `config/experiments/reciprocity_calibration.yaml` (seed 20260808, λ = 1.2)
- Expected / observed: informal expectation was ℛ ∈ [0,1]; matching pennies reads ℛ = 1.2. The definition ‖χ−χᵀ‖/‖χ+χᵀ‖ is a norm *ratio*, unbounded above — values > 1 mean the antisymmetric (circulating) part of the equilibrium response dominates the symmetric part.
- Chase status: resolved (definitional, not a bug). RPS-3 reads 0.69, RPS-5 reads 0.43, MP reads 1.20 — the 2×2 tangent space (1-D per player) makes MP the most purely rotational response geometry available.
- Resolution / follow-up: docs corrected to state ℛ ∈ [0, ∞). Worth watching in the α-sweep whether ℛ > 1 is exclusive to (near-)pure harmonic games with minimal action spaces — if a *mixed* game ever reads > 1 at moderate α, that is structure worth understanding.

## F-0002 — ℛ magnitude scales with λ (only the zero/nonzero reading is λ-free)

- Date: 2026-08-08
- Instrument / experiment: `reciprocity_defect`, red-team probe during unit review (matching pennies, λ ∈ {0.5, 1, 2, 4})
- Expected / observed: ledger/docs briefly claimed "ℛ is λ-free"; observed ℛ(λ) ≈ ∝ λ on matching pennies (0.5, 1.0, 2.0, ... at λ = 0.5, 1.0, 2.0). Consistent with χ_T = (I − S_T B_T)⁻¹S_T with S ∝ λ: the antisymmetric part enters at O(λ²) against a symmetric O(λ) base, so the ratio grows ≈ linearly at small λ.
- Chase status: resolved (theory understood; docs/ledger corrected). The λ-free property is Result 2's *symmetry* statement: ℛ = 0 ⟺ potential, at every λ.
- Resolution / follow-up: empirical use of ℛ levels (Stage 3 pass-through work) must hold λ fixed or co-report it; only the zero test is λ-robust. Candidate for a normalised variant ℛ/λ_normalised if cross-λ comparability becomes needed.

## F-0003 — EPR, ℛ and α co-move almost perfectly on random 3×3 families

- Date: 2026-08-08
- Instrument / experiment: `thermo_read` + `reciprocity_defect` via `experiments/dynamics_calibration.py`
- Config: `config/experiments/dynamics_calibration.yaml` (seed 20260808, λ = 1.2, 100×10 games)
- Expected / observed: C1 predicted co-movement but not its tightness; observed ρ(EPR, ℛ) = 0.993 — the dissipation of the *joint-profile Markov chain* and the asymmetry of the *equilibrium response matrix* are near-interchangeable orderings on these families, despite being computed from entirely different objects (generator vs resolvent).
- Chase status: superseded by F-0004 — the marginal agreement is α-driven confounding, discovered by red-team stratification.
- Resolution / follow-up: see F-0004.

## F-0004 — The meters DECOUPLE at high α: within-level ρ(EPR, ℛ) reverses sign

- Date: 2026-08-08
- Instrument / experiment: stratified re-analysis of `chain_comovement` (red-team objection during dynamics.exact review; independently verified same day, then made a permanent artifact metric)
- Config: `config/experiments/dynamics_calibration.yaml` (seed 20260808, λ = 1.2, 100 games × 10 α levels)
- Expected / observed: C1 expected tight co-movement of dissipation and reciprocity defect. Observed: the marginal ρ(EPR, ℛ) = 0.993 is driven by α. **Conditional on α**, coupling is strong (+0.80..+0.88) for α ≤ 0.65, degrades through 0.75 (+0.61) and 0.85 (+0.32), and **reverses to −0.355 at α = 0.95**: among near-pure-harmonic games, higher reciprocity defect associates with LOWER dissipation.
- Chase status: **resolved by F-0007** (mechanism experiment, same games, paired design). Half the hypothesis held: ℛ's high-α variation IS denominator-driven (ρ(ℛ, 1/‖χ+χᵀ‖) = 0.993 at α = 0.95). But the numerator fix fails — see F-0007.
- Resolution / follow-up: (1) test the hypothesis by tracking numerator and denominator of ℛ separately along α; (2) check whether an EPR-normalised or numerator-only variant restores coupling at high α; (3) verify on N=3 and larger m; (4) this belongs in p3_noneq as a result, with C1's falsifier status stated plainly.

## F-0005 — Blotto is mixed, not pure harmonic: α = 0.69 on the symmetric budget-3 game

- Date: 2026-08-08
- Instrument / experiment: `alpha` via `experiments/blotto_calibration.py` (seed 20260808)
- Expected / observed: DOMAINS v1 called Blotto "non-potential (harmonic-ish)"; the meter reads α = 0.694 (budget 3, k = 3) and 0.711 (budget 4) — a genuine ~30% potential component survives normalisation despite the game being zero-sum. Zero-sum ⇏ harmonic-pure: the normalised Blotto game has real gradient structure (allocation dominance relations) alongside its circulation.
- Chase status: parked (consistent with theory; RPS by contrast reads exactly 1). Worth revisiting when the α > 0 anchor is used quantitatively: the bracket's right end sits at ~0.7, not 1.0.
- Resolution / follow-up: use RPS for the α = 1 extreme and Blotto for a *realistic* high-α strategic system; note in p1_instruments.

## F-0006 — The criticality wedge: potential games ESCAPE criticality at high λ; the supercritical frontier descends with α

- Date: 2026-08-08
- Instrument / experiment: `experiments/phase_map.py` (seed 20260808; 9 λ × 11 α × 5 games, median surface in `phase_map_surface.json`)
- Expected / observed: naively ρ(SB) ∝ λ (since S ∝ λ). Observed: **on α = 0 the median ρ is non-monotone** — 0.07 → 0.51 (λ = 1.7) → 0.0 (λ = 15). Mechanism (checked against theory, not just eyeballed): as λ → ∞ on a potential game, the principal branch concentrates on a strict equilibrium, C = diag(σ) − σσᵀ → 0 exponentially, so S = λC → 0 despite the λ factor — the meter escapes criticality through equilibrium concentration. At α = 1, mixed equilibria keep C bounded away from zero and ρ grows without bound (6.9 at λ = 15). In between, a **supercritical wedge** opens: the *median* game first crosses ρ = 1 at α = 0.5 (λ ≥ 8.5); a 0.2 fraction of games crosses already at α = 0.4; the frontier descends monotonically with α (λ_c ≈ 3 by α = 0.8). (Onset wording corrected per red-team: median-onset is 0.5, not 0.4.) **Caveat**: inside the wedge (ρ ≥ 1) the resolvent is near-singular, so ℛ medians there are magnitude-unreliable — direction only.
- Chase status: chasing. (1) Is the α = 0 peak location λ ≈ 1.7 tied to the payoff scale (scale = 2.0 here — test the folding); (2) is the wedge boundary a clean curve λ_c(α) (finer grid); (3) does the wedge boundary coincide with where within-level ℛ–EPR coupling degrades (F-0004's α ≈ 0.75 is suspiciously inside the wedge)?
- Resolution / follow-up: finer sweep along the frontier; p3_noneq's phase-map section; the Lab's live phase panel should draw the wedge.

## F-0007 — The response layer and the dissipation layer are distinct observables (H1 refuted, H2 confirmed)

- Date: 2026-08-08
- Instrument / experiment: `experiments/decoupling_mechanism.py` (paired design: the identical seeded families as `chain_comovement`, seed 20260808, λ = 1.2)
- Expected / observed: F-0004's working hypothesis — itself proposed *post hoc* to explain the observed decoupling, and tested here as a genuine prediction — predicted the numerator ‖χ−χᵀ‖ would keep tracking EPR at high α (only the ratio's denominator misbehaving). **Observed: H2 confirmed — at α = 0.95, ρ(ℛ, 1/‖χ+χᵀ‖) = 0.993, rising monotonically from ≈0 at low α: ℛ's within-level variation at high α is entirely the shrinking symmetric response. H1 refuted — the numerator ALSO decouples (ρ(num, EPR) = −0.37 at α = 0.95, from +0.77..0.86 at α ≤ 0.65).** No renormalisation of the response matrix recovers the dissipation ordering in the near-harmonic regime.
- Chase status: resolved as a structural fact, and it is the *better* result: χ^eq is a local derivative at the QRE point; EPR is a global functional of the stationary flux of the joint-profile chain. They co-vary exactly while a potential component modulates both (α ≤ 0.65) and part company when it vanishes. **ℛ remains the potentiality *test* (= 0 iff potential, exact at every λ); EPR remains the dissipation *gauge*; neither substitutes for the other across the harmonic regime.** Negative result for the corrected-meter idea, written down plainly.
- Resolution / follow-up: p1_instruments states the instrument-scope table (which meter answers which question, with the α-regime of validity); p3_noneq gets the mechanism section. Open: why num ~ 1/den within-level at α = 0.95 (near-perfect rank agreement of numerator with inverse denominator) — likely overall χ-scale variation; check by conditioning on ‖χ‖_F.

## F-0008 — First real-data reading: DAM at-null; RTM escapes its null on the LOW side (REVISED after red-team)

- Date: 2026-08-11 (revised same day — the original "certified null" claim was retracted on red-team review)
- Instrument / experiment: `kld_epr` on discretized/embedded price series via `experiments/electricity_reading.py` (unit domains.electricity)
- Config: `config/experiments/electricity_reading.yaml` (seed 20260811; SP15; DAM hourly 35d n=840; RTM 5-min 14d n=4032; 6 bins; 200 surrogates each of FT and AAFT)
- Expected / observed: expected the diurnal ramp to register as irreversibility. Observed, with the CORRECT null (amplitude-adjusted FT — plain FT Gaussianizes a kurtosis-132 marginal and was rejected by red-team): **DAM hourly prices sit INSIDE the AAFT null band** (0.0447 vs band [0.034, 0.075]) — consistent with a linear time-reversible process at this n/resolution/embedding. **RTM 5-minute prices escape the null band on the LOW side** (0.0474 vs AAFT q01 = 0.239, 5–6× below the median): NOT a detection and NOT a certified null — the linear+marginal surrogate class cannot reproduce the data's Δ-sign persistence (real RTM prices flip direction far less than any linear process with their spectrum+marginal), so no adequate null exists within the classes tried.
- Also established (proven in unit tests, then observed on data to 4 decimals): price-VALUE discretization is structurally blind to loop irreversibility; plain-shuffle nulls fail twice (persistence; iid-embedding asymmetry ~0.15 nats); plain-FT nulls fail on heavy tails. Three wrong nulls caught, each now a permanent guard.
- Chase status: chasing. (1) Constrained surrogates that match Δ-sign persistence (IAAFT with iteration, Markov-matched or block surrogates) to get a bracketing null for RTM; (2) the RTM persistence anomaly itself — WHY do real-time prices ramp so much more persistently than their linear image? (candidate: dispatch inertia / ramp constraints — possibly the irreversibility hiding in a statistic the KLD at k=1 cannot see); (3) finer bins, longer windows, spike-conditioned reads.
- Resolution / follow-up: the honest verdicts are "DAM: at-null", "RTM: no detection, null construction open". p3_noneq data section blocked on chase item (1).

## F-0009 — First positive real-data detection: DAM hourly prices violate pair-level detailed balance (diurnal-loop signature)

- Date: 2026-08-11
- Instrument / experiment: third null class added to `experiments/electricity_reading.py` — the REVERSIBILIZED-MARKOV surrogate (fit the embedded chain's pair counts, symmetrize the flux (C+Cᵀ)/2, simulate the detailed-balance chain with identical persistence, re-read)
- Config: as F-0008 (seed 20260811, 200 surrogates per class)
- Expected / observed: F-0008's chase item 1 asked for a null that brackets persistence. Observed: **DAM hourly reading 0.0447 nats/h EXCEEDS the reversible-Markov null q99 = 0.0291 (markov_detected=1)** — the observed pair-flux asymmetry of the (price-bin, Δ-sign) chain is beyond reversible-chain chance at matched persistence: a detection of the diurnal loop's irreversibility at hourly resolution, previously masked by the spectral nulls' inflated flip-rate bias. **RTM 5-min is genuinely at-null in the matched class** (0.0474 inside [0.002, 0.079]) — at that sampling rate dwell dominates and k=1 sees nothing; k>1 or coarser Δt is the follow-up there.
- Caveats (stated before anyone else states them): the Markov null is a parametric bootstrap under the reversibility restriction of the FITTED first-order chain — it tests pair-level (k=1) detailed balance of the embedding, not full-process reversibility; and this finding has NOT yet had its own adversarial review (the unit gate closed on F-0008's revised wording; F-0009 is a chase resolution logged pending red-team).
- Chase status: chasing — (1) independent red-team verification of the Markov-null construction; (2) RTM at k=2 / coarser Δt; (3) effect size in physical terms (nats/day of the diurnal cycle ≈ 1.1) for p3_noneq's data section.
- Resolution / follow-up: reconciles the three null classes into one picture: FT (wrong marginal), AAFT (wrong flip rate), reversibilized-Markov (matched persistence — the sharp test). p3_noneq data section can now be drafted around DAM-detected / RTM-open.

### F-0009 addendum — red-team verification + mandatory temporal stratification (same day)

- **Red-team verdict: granted-conditional.** Verified numerically: the reversibilized chain satisfies detailed balance exactly (violations 0 at 1e-10; stationarity residual 3e-17); false-positive rate on truly reversible data 1/30 (consistent with α=0.01); detection robust to surrogate seeds (5/5), n_price_bins ∈ {2,3,4} (3/3), tie-breaking flip (stat moves 0.9%), and Bonferroni across 2 markets × 3 null classes (empirical p < 0.01, q99 at α/2 still exceeded); higher-order (order-2) reversible structure does not explain the pair statistic.
- **Blocking condition (subsample fragility) resolved via weekly stratification** (5 × 7d blocks, per-block nulls): every week's statistic exceeds its null MEDIAN (ratios 1.63, 1.52, 3.03, 9.10, 4.17 — the signal has consistent sign throughout), but per-week significance concentrates in the second half (weeks 4–5: p ≤ 0.005; week 3: 0.035; weeks 1–2 underpowered at n=168). Leave-one-week-out: detection survives removal of weeks 1, 2, or 4 but not 3 or 5. **Explicit acknowledgment per red-team condition (B): the July-2026 detection is statistically driven by the high-ramp second half of the window — physically sensible (irreversibility concentrates where the diurnal drive is strongest, summer scarcity ramps) and stated as such wherever the number is used.**
- **Garden-of-forking-paths note (red-team O-2)**: the Markov null was chosen after two spectral nulls failed; mitigation is the prominently-reported empirical p-value (<0.01) and this documented selection history.
- p3_noneq data section may now cite F-0009 with the concentration caveat verbatim.

### F-0009 addendum 2 — RTM k=2 probe (chase item closed)

- RTM 5-min vs its reversibilized-Markov null at k=2: stat 0.0833 vs null [med 0.0591, q99 0.1176] — still no detection (and k=1 re-confirmed: 0.0474 vs q99 0.1026). The RTM at-null verdict is robust to block order at this resolution; remaining RTM avenue is coarser Δt aggregation (30–60 min), queued low-priority.

### F-0009 addendum 3 — bidding-oracle pipeline: stylised model REJECTED by the data

- The electricity plugin's five-object contract is complete (uniform-price BiddingOracle, offer-ladder grid, exact-linear offer-shift conjugate field, CAISO loader, learn spec; α of the calibrated auction is mixed, not an anchor — unit-tested). The conditional λ̂ pipeline (match QRE clearing-price dispersion to observed) returned the RIGHT kind of answer on July-2026 SP15: **model rejected** — the stylised duopoly's dispersion ceiling (10.4 $/MWh, its λ→0 maximum) sits 40% below the observed spike-driven std (16.8), so no λ is reported (`electricity_lambda.json`, model_rejected=1, λ̂=null). Scarcity spikes exceed what 2-agent undercutting can generate; a richer supply model (capacity constraints / N>2 / demand uncertainty) is the follow-up before any market-λ claim.

## F-0010 — The F-0004 reversal is λ-dependent: universal COLLAPSE, conditional SIGN FLIP

- Date: 2026-08-11
- Instrument / experiment: `experiments/frontier.py` crossover sweep (unit science.frontier; seed 20260811; α∈{0.55..0.95} × λ∈{0.8,1.2,2.0} × 60 games at 3×3, plus 4×4 at λ=1.2)
- Expected / observed: the initial criterion (ex-ante hypothesis, written before the run but not externally locked — not 'pre-registered' in the strict sense) expected F-0004's negative within-level ρ(EPR,ℛ) at α=0.95 in ≥3 of 4 conditions — **it failed (2/4)**, and the failure is the finding. What IS universal: the COLLAPSE of coupling at α=0.95 (from ~0.90 at α≤0.65 to ≤0.25 in every condition, both sizes). The SIGN goes negative only as λ grows: +0.25 (λ=0.8) → +0.03 (λ=1.2) → −0.23 (λ=2.0); and −0.26 at 4×4/λ=1.2. F-0004's −0.355 (different seeds, 100 games) sits within this gradient.
- Reading: the structural fact (F-0007's local-derivative-vs-global-flux decoupling) shows up first as decorrelation; the negative sign is λ-DEPENDENT across the three λ tested (each individual sign within ~2 null-SD at n=60 — suggestive, not established) — plausibly the supercritical wedge's doing (λ=2 at α=0.95 is deep inside it, where ℛ magnitudes are resolvent-dominated). The 4×4 negativity at λ=1.2 suggests size also pushes toward reversal.
- Criterion revision (documented, not silent): the artifact's pass criterion is revised from "reversal in ≥3/4 conditions" to "collapse (corr < 0.35 at α=0.95) in ALL conditions, with the sign pattern recorded as data"; the original criterion and its failure are in this entry and the artifact notes.
- Chase status: parked with a sharp question for p3: is sign(corr at α=0.95) a function of distance INTO the wedge (ρ−1) rather than of λ per se? One regression on existing per-game data would answer it.
- Resolution / follow-up: p3_noneq decoupling section gains the λ-gradient; C1's crossover row in claims.md updated.

### F-0010 addendum — the wedge-depth hypothesis REFUTED (the last open chase of this run)

- Date: 2026-08-11
- Test: pooled per-game (ρ(SB), EPR, ℛ) at α=0.95 across λ∈{0.8,1.2,2.0}, 60 games each (same seeds as the crossover sweep).
- Observed: **0 of 180 games are supercritical** at these λ (consistent with the measured frontier: λ_c(0.80)≈3.0 descending ⇒ λ_c(0.95) still above 2). Yet the within-level sign flips with λ across exactly these conditions (+0.25 → +0.03 → −0.23). Conclusion: the F-0004/F-0010 decoupling and its λ-dependent sign occur ENTIRELY OUTSIDE the supercritical wedge — wedge depth is not the driver, and F-0006's "suspicious coincidence" (decoupling regime sitting inside the wedge on the coarse map) was a coincidence of the coarse grid's λ range, not a mechanism. The remaining candidate driver is λ-dependent equilibrium concentration itself (C shrinking, reweighting the χ scale) — a p3 discussion point, not a claim.
- Methods note: a pooled-across-λ Spearman (+0.89) was computed and DISCARDED as λ-confounded (both meters grow with λ) — the same confound F-0004 taught us marginally over α; recorded so nobody re-runs the wrong cut.

## F-0011 — The reciprocity meter's first empirical read: ℛ ≈ 0.001 on real pass-through data, exactly where theory demands zero

- Date: 2026-08-11
- Instrument / experiment: `experiments/pricing_reading.py` (unit domains.pricing) on the PI's HF Dominick's canned-soup panel (`qbz506/dreamprice-dominicks-cso`, CC-BY-NC-4.0, Kilts Center origin; ~500K store×UPC×week rows, a random subsample handled gap-tolerantly via brand-level store-week indices)
- Design + ex-ante prediction (in the config BEFORE the run): cross-brand wholesale-cost pass-through between Campbell and Progresso, regular prices only, two-way (store, week) demeaning, cluster bootstrap over 86 stores. Because ONE retailer prices both brands (category management), a single-objective optimizer must respond symmetrically — prediction: small ℛ, asymmetry CI covering zero.
- Observed: own pass-through 1.07 / 0.97 (textbook); cross terms 0.0028 / 0.0005; asymmetry CI [−0.005, +0.010] ∋ 0; **ℛ_empirical = 0.0011 [0.00005, 0.0050]**. The prediction is confirmed: real category pricing reads potential-like, and N2's "ℛ estimable from pass-through asymmetry without payoffs" is now an executed measurement, not a promise. C2's first anchor: ℛ_emp sits at the potential end of the synthetic bracket (Sioux 0 / Blotto 0.12 / RPS 0.69), consistent rather than orders off.
- Companion scan (same artifact set): Edgeworth-cycle test on 30 stores' weekly category price indices vs the reversibilized-Markov null — **0/30 detections** (expected false positives 0.3): weekly category indices are at-null; the Maskin–Tirole signature, if present, lives at item-level/finer timing (and the HF subsample's week gaps push toward reversibility — detections would have been conservative, non-detections are not).
- Bug caught before review: the first run's χ row ordering swapped own/cross slots (giving a spurious ℛ = 0.54); caught by economic sanity inspection (both prices "loading on one brand's cost"), fixed, rerun — recorded here so the artifact history is interpretable.
- Red-team (2026-08-11, granted after one condition): the mechanical-correlation attack (COST derived from PRICE) was tested numerically and CLEARED — residual price–cost correlation after demeaning is 0.94–0.96, margin CV 0.26–0.35 (identification comes from idiosyncratic margin movement, not the accounting identity), and the cross terms use the OTHER brand's cost so they are never mechanically linked. Blocking condition — the bootstrap demeaned once and resampled demeaned rows — closed by re-implementing the standard protocol (re-demean within every resample, duplicated stores relabelled as distinct clusters): ℛ CI moved [0.0001, 0.0050] → [0.00005, 0.0050], asymmetry CI unchanged to 4 decimals. Also on record: the "ex-ante" config comment is same-session, not externally locked (the honest wording is "stated in config before the run"), and single-retailer symmetry is exact only under a symmetric demand Jacobian — the observed near-zero cross terms make ℛ ≈ 0 robust to that caveat. Edgeworth series lag-1 autocorrelation ≈ 0.57, so the at-null verdict is about persistent price dynamics, not basket-composition noise.
- Chase status: parked with two sharp follow-ups: (1) the TRUE multi-retailer reciprocity question needs cross-CHAIN data (Dominick's vs competitors) — this dataset cannot ask it; (2) item-level Edgeworth at finer resolution once a non-subsampled panel slice is used.
- Resolution / follow-up: p1's instruments section gains the empirical ℛ row; C2 updated.

## F-0012 — The driving cost inverts across the family: potential games pay per change, harmonic games pay per unit time

- Date: 2026-08-12
- Instrument / experiment: `experiments/protocol_reading.py` (unit thermo.protocols) — Hatano–Sasa housekeeping/excess split + stepwise λ-quench (0.5→3.0) across the α family (coordination↔matching-pennies mix, exact dense 2×2 joint chains).
- Pre-registration status (stated honestly, red-team O-3 follow-through): the config with P1–P3 was WRITTEN before the run (P1 excess ⟨Y⟩ ~1/K under refinement; P2 housekeeping ≈ 0 at α=0, monotone in α, duration-linear; P3 ⟨e^{−Y}⟩ = 1 machine-precision), and a pre-registration commit was attempted before the experiment — but that commit ABORTED on a pre-commit hook failure and did not land; the red-team caught the gap (config staged, not committed, when results existed). So this run is file-mtime-ordered, not commit-ordered. The predictions were not adjusted post hoc, but the audit trail cannot prove that; recorded as a process failure, and the standing rule is upgraded: verify `git log` shows the config commit landed before running.
- Observed: **all three PASS** — refinement slopes −1.02/−0.94 (the 1/K law), housekeeping 0 → 0.72 → 3.39 → 8.00 → 12.38 nats across α with duration-linearity R² = 1.000 (burn rate 0.774 nats/time at α=0.95), max IFT error 0.0; sampled-path IFT CI [0.9997, 1.0002] ∋ 1 over 3000 exact-kernel trajectories.
- **The honest deviation**: P1's prose expected excess "the same order across α". Wrong — excess COLLAPSES with α (0.036 → 2.8×10⁻⁵, three orders): as the mix approaches matching pennies the NESS becomes λ-insensitive (uniform at every λ), so there is almost nothing to relax when λ moves. The refinement-law pass criterion was unaffected; the wording is corrected here rather than papered over.
- Reading: the cost of driving a strategic system inverts across the family. Potential games pay only for CHANGE (excess ∝ how coarsely you step λ; quasi-static driving is asymptotically free) and nothing to exist. Near-harmonic games pay almost nothing for change but a constant HOUSEKEEPING rent per unit time just to hold their NESS — at α=0.95 the housekeeping/excess ratio at the base protocol is ~4.4×10⁵. "You cannot drive a whirlpool slowly for free" — worse: the whirlpool charges rent whether you drive it or not.
- **Mechanism: OPEN (red-team O-2)**. The first-pass explanation — "the near-harmonic NESS is λ-insensitive, so there is nothing to relax" — was REFUTED by the red-team's probe: on an asymmetric α=0.95 mix whose NESS genuinely moves with λ (‖Δπ‖ ≈ 7×10⁻³ across the ramp), excess still collapses ~290×. The collapse is real; its driver is not yet identified (candidate: λ-dependent equilibrium concentration re-weighting the relaxation spectrum). Chase items: mechanism isolation, larger action spaces, asymmetric harmonic sources.
- Also on record (red-team O-1): the exact-transfer ⟨e^{−Y}⟩ = 1 output is a telescoping algebraic identity for stepwise protocols — it cannot catch kernel bugs; the correctness check is the independent sampled path. And the sampled check at α=0.95 has ⟨Y⟩ ~ 6×10⁻⁵, so its CI-brackets-1 verdict is low-power there; a moderate-α (0.5) sampled check with real power is in the artifact.
- Resolution / follow-up: p3 gains the protocol section; the split is the estimator-facing groundwork for Hatano–Sasa-style estimation on data.
