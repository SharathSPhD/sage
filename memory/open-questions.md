# Open questions — the live queue

Working questions, not hypotheses to adjudicate. Move answered ones to findings.md or claims.md.

1. **Is ℛ(α) smooth?** Or structured — thresholds, plateaus, non-monotonicity? Is Spearman ρ(ℛ, α) > 0.9 across families, or only within them? (α-sweep, Stage 5.)
2. **Does N3 hold numerically?** No cycles at α = 0 anywhere in (λ, N, m). A cycle at α = 0 is either the tangent-space bug (check first) or a headline finding. (Ledger row N3 reverts hard if so.)
3. **Where do the four λ estimators (MLE, CCP, dispersion/FDT, hierarchical) agree**, and does their divergence track anything interpretable (non-potentiality? misspecification? promotion regimes)? The dispersion-vs-MLE gap is itself a detailed-balance diagnostic.
4. **Is round-level Blotto experimental data actually obtainable** (Chowdhury–Kovenock–Sheremeta; Arad–Rubinstein; Duffy–Matros)? Check before promising empirics; synthetic + published aggregates is the fallback. (DOMAINS v1 §4.2.)
5. **Does the Kronecker Hodge transform scale near-linearly in practice** (the §1.2 claim) up to N=4–5, m=30–50 on this hardware? Benchmark early.
6. **ERCOT offer-curve aggregation**: which action definition (per-unit steps vs portfolio aggregates) is defensible against the supply-function-equilibrium literature, and how sensitive are readings to it?
7. **Does ℛ from Dominick's pass-through asymmetry land anywhere near synthetic games at comparable α** (claim C2)? Robust to demand specification?
8. **Do the meters read exactly zero on TNTP networks** with computed SUE flows (population engine)? Any nonzero is a bug by definition there — the calibration standard.
9. **Population engine effort**: is the 30–40%-of-Engine-1 estimate right? Timebox; fallback is congestion as a large finite-N approximation (loses exactness only).
10. **What does the spectrum of SB look like across real domains** — do empirical games sit near criticality or far from it?
