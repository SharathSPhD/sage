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
