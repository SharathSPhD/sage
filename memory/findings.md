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
