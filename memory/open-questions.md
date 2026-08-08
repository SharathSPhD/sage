# Open questions — the live queue

Working questions, not hypotheses to adjudicate. Move answered ones to findings.md or claims.md.

1. **Is ℛ(α) smooth?** PARTIALLY ANSWERED 2026-08-08: rank-monotone along α (ρ = 0.982, n = 2000); genuine cross-game structure at equal α (RPS-3 ≠ RPS-5); ℛ–EPR coupling reverses within-level at α ≥ 0.85 (F-0004). Remaining: functional form, λ-dependence of the surface (phase map running), larger shapes.
2. **Does N3 hold numerically?** SPECTRAL LEG ANSWERED 2026-08-08: SB spectrum exactly real on all potential games across λ ∈ {0.5..10} (`spectrum_reality.json`). Remaining: dynamic (trajectory-level) cycles at α = 0 with the estimator layer.
3. **Where do the four λ estimators (MLE, CCP, dispersion/FDT, hierarchical) agree**, and does their divergence track anything interpretable (non-potentiality? misspecification? promotion regimes)? The dispersion-vs-MLE gap is itself a detailed-balance diagnostic.
4. **Is round-level Blotto experimental data actually obtainable** (Chowdhury–Kovenock–Sheremeta; Arad–Rubinstein; Duffy–Matros)? Check before promising empirics; synthetic + published aggregates is the fallback. (DOMAINS v1 §4.2.)
5. **Does the Kronecker Hodge transform scale near-linearly in practice** (the §1.2 claim) up to N=4–5, m=30–50 on this hardware? Benchmark early.
6. **ERCOT offer-curve aggregation**: which action definition (per-unit steps vs portfolio aggregates) is defensible against the supply-function-equilibrium literature, and how sensitive are readings to it?
7. **Does ℛ from Dominick's pass-through asymmetry land anywhere near synthetic games at comparable α** (claim C2)? Robust to demand specification?
8. **Do the meters read exactly zero on TNTP networks?** ANSWERED 2026-08-08: ℛ = 5.7e-17, DF defect 0, KKT 7e-15 on Sioux Falls (units population.core / domains.congestion).
9. **Population engine effort**: ANSWERED 2026-08-08 — well under the 30–40% estimate (one session: routing games, Newton SUE, susceptibility, TNTP). The k-shortest route-set generation, not the engine, is where future cost lives (full Sioux Falls, Anaheim).
10. **What does the spectrum of SB look like across real domains** — do empirical games sit near criticality or far from it?
