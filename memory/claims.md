# Claims ledger

Every scientific claim, its confidence tier, evidence, and status. **Owner: theory-verifier** — nobody else changes tiers. Tiers: `exact` (proved identity; implement, cite, never claim) · `derived` (proved in-house and/or checked; ours to claim with care) · `conjectured` (argued, not established; must state its falsifier) · `speculative` (exploratory framing only).

Source: PROGRAMME v3 §3; updated by the prior-art sweep of 2026-08-08 (`literature-nearest-live-work.md`). Unit ids link claims to gates (stage0 seeded this ledger).

## Known results — `exact`. Cite, never claim.

| ID | Claim | Citation |
|---|---|---|
| K1 | Logit response = argmax of E[U] + λ⁻¹H(σ) (entropy-regularised best response; Gibbs variational principle) | Fudenberg–Iijima–Strzalecki, *Econometrica* 2015 |
| K2 | Log-partition ψ is the CGF: ∇ψ = λσ, ∇²ψ = λ²C with C = diag(σ) − σσᵀ | Williams–Daly–Zachary; exponential family |
| K3 | Exact potential Φ ⟹ Glauber/logit dynamics reversible, π ∝ e^{λΦ}, J* = 0, EPR = 0 | Blume 1993; Monderer–Shapley 1996 |
| K4 | Externality symmetry (B = Bᵀ on the appropriate space) ⟺ **full** potential game; on the tangent space / normalised game ⟺ potential game. **The full-vs-effective distinction is mandatory** — full symmetry is sufficient but not necessary for potentiality of the raw game (arXiv:2405.07224 Lemma C.2, Example C.2) | Sandholm 2001, 2010 Ch.3; Balduzzi et al. ICML 2018; Candogan et al. 2011 |
| K5 | Hodge/flow decomposition: 𝒢 = 𝒢_pot ⊕ 𝒢_harm ⊕ 𝒢_nonstrat, orthogonal; nonstrategic components do not affect QRE | Candogan–Menache–Ozdaglar–Parrilo, *MOR* 2011 |
| K6 | λ has a rational-inattention foundation: inverse shadow price of Shannon information | Matějka–McKay *AER* 2015; Fosgerau et al. *IER* 2020 |
| K7 | Partial susceptibility χ^part = λC (static FDT, opponents frozen) | K2 + differentiation |
| K8 | Logit route choice in congestion games = Fisk (1980) SUE, convex potential Σₐ∫cₐ + λ⁻¹Σ xₐlog xₐ | Fisk 1980; Beckmann 1956; Rosenthal 1973. **Implemented and verified 2026-08-08** (units population.core, domains.congestion): ∇Beckmann ≡ route costs to 1e-12 by autodiff; KKT to 7e-15; ℛ = 5.7e-17 on real Sioux Falls data |

**History note:** v2 of the programme claimed the symmetry⟺potentiality characterisation as new. It is textbook (K4). Downgraded during the v3 sanity check — the precedent for how this ledger works.

## Our results — `derived`

| ID | Claim | Evidence | Status |
|---|---|---|---|
| R1 | Strategic resolvent: χ^eq = (I − SB)⁻¹S on the tangent space | Total differentiation of the fixed point; PROGRAMME v3 §3.3 | Likely folklore in some form — **cite defensively, don't claim**. **Numerically verified 2026-08-08**: matches central finite differences to 1.3e-8 on 50 random games (`chi_fd_agreement.json`); implicit-diff custom VJP reproduces the same resolvent to 1e-8 on 20 games (`implicit_chi_agreement.json`, unit solve.advanced); fixed-λ profiles match pygambit's homotopy to 1e-8 (`gambit_agreement.json`); the arclength branch tracer (unit solve.branch) passes through fixed-λ points to 1e-7 along the correspondence (`branch_agreement.json`) |
| R2 / N1 | **Reciprocity transfer**: χ^eq symmetric ⟺ S(B − Bᵀ)S = 0 ⟺ (full support) B = Bᵀ on T ⟺ zero harmonic component of the normalised game. Strategic feedback neither creates nor destroys reciprocity | 3-line proof, PROGRAMME v3 §3.3; prior-art sweep 2026-08-08 found **no prior statement** | Novel in form; trivial algebra, operationally important. **Numerically verified 2026-08-08** (unit finite.response.reciprocity): ℛ ≤ 9e-17 on 5 exact potential games; ℛ ∈ [0.43, 1.2] on harmonic games; Spearman ρ(ℛ, α) = 0.982 over 2,000 games (artifacts in benchmarks/results). ℛ is unbounded above — findings F-0001 |
| N2 | ℛ = ‖χ^eq − χ^eqᵀ‖_F / ‖χ^eq + χ^eqᵀ‖_F is an **observable proxy for harmonic content**, estimable from cross-agent cost-shock pass-through asymmetry without knowing payoffs. **λ-free as a symmetry statement only** (whether ℛ = 0 is λ-independent; the magnitude scales with λ — findings F-0002, red-team O-1) | Consequence of N1; prior-art sweep found no prior use of response-matrix asymmetry as a harmonic proxy | Novel framing. The λ-free property is the lead answer to HHK. Empirical estimator: Stage 3 |
| N3 | No Hopf bifurcation of logit dynamics in full potential games: B = Bᵀ ⟹ SB ∼ S^{1/2}BS^{1/2} symmetric ⟹ real spectrum ⟹ no complex pair crossing | Trivial similarity argument. **Upgraded conjectured → derived** by theory-verifier (prior-art sweep 2026-08-08): argument sound, not previously written down for logit; Hommes–Ochea 2012 (Hopf in RPS under logit) supports the contrapositive | **Numerically verified 2026-08-08** (artifact `spectrum_reality.json`): SB spectrum exactly real on all 5 potential games across λ ∈ {0.5..10}; visibly complex (min imag 0.29) on harmonic games. Full α-sweep dynamics check still to come with the thermo layer; a cycle at α = 0 would revert this row hard |

## Conjectured — must state their falsifier

| ID | Claim | Falsifier | Status |
|---|---|---|---|
| C1 | The chain cycling → non-potential → broken reciprocity → positive dissipation holds *tightly* (the four meters co-move monotonically along α) | A regime in the α-sweep where the meters decouple (e.g. ℛ rises but EPR stays ≈ 0, or cycling without dissipation jump) | **Falsifier PARTIALLY REALISED 2026-08-08** (see also the phase surface, unit science.phase_map: F-0004's decoupling regime sits inside the supercritical wedge of F-0006) (unit dynamics.exact, artifact `chain_comovement.json`, findings F-0004): marginally, ρ(EPR, α) = 0.990 and ρ(EPR, ℛ) = 0.993 — but stratified by α, within-level ρ(EPR, ℛ) falls from +0.88 (α ≤ 0.65) to **−0.355 at α = 0.95**. The chain co-moves *along* α but the meters decouple, and reverse, *within* the near-harmonic regime. C1 as stated ('holds tightly') is FALSE conditionally at high α; retained as `conjectured` in the refined form 'marginal co-movement along α, conditional decoupling above α ≈ 0.75'. Discovered via red-team stratification |
| C2 | ℛ measured from real pass-through asymmetry (Dominick's/ERCOT) is comparable in magnitude to synthetic games at matched α | ℛ_empirical persistently orders-of-magnitude off the synthetic curve across demand specifications | Open; Stage 3+. Synthetic α > 0 anchor now in place (unit domains.blotto: α = 0.69, ℛ = 0.12 at λ = 1.5 on budget-3 Blotto — `blotto_readings.json`) |

## Speculative — not for user-facing docs

| ID | Claim | Note |
|---|---|---|
| S1 | Sparse/entmax quantal response fits 9-ending price concentration better than logit | Hypothesis from research-main.md; test if it ever matters, not assumed |
| S2 | λ vs collusion separability via (mean, dispersion) signature + ℛ + dissipation persistence | Separation will be partial; report what separates and what doesn't (PROGRAMME v3 §4.2) |
