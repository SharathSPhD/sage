# p1_instruments — results digest

Every number in the manuscript, its value, and its regeneration path. Build:
`uv run python papers/p1_instruments/make_figures.py && cd papers/p1_instruments && pdflatex p1_main.tex (x2)`.
All artifacts regenerate via `make reproduce` (seed 20260808).

| Manuscript statement | Value | Artifact (benchmarks/results/) | Gate unit |
|---|---|---|---|
| Gibbs agreement, congestion n=2/3, λ∈{0.7,1.2,2} | 2.5e-16 | gibbs_agreement.json | dynamics.exact |
| EPR, \|J*\| on potential games | < 1e-12 | equilibrium_reads_zero.json | dynamics.exact |
| ℛ on 5 exact potential games | ≤ 9e-17 | reciprocity_potential.json | finite.response.reciprocity |
| ℛ on RPS-3 / RPS-5 / MP | 0.69 / 0.43 / 1.20 | reciprocity_harmonic.json | finite.response.reciprocity |
| χ^eq vs finite differences (50 games) | 1.3e-8 | chi_fd_agreement.json | finite.response.reciprocity |
| Real SB spectrum on potential games, λ≤10 | rel. imag = 0 | spectrum_reality.json | finite.response.reciprocity |
| Spearman ρ(ℛ, α), n=2000 | 0.982 [bootstrap CI in artifact] | reciprocity_alpha_sweep.json | finite.response.reciprocity |
| ρ(EPR, α)=0.990; ρ(EPR, ℛ)=0.993 marginal; within-level +0.88→−0.36 | see artifact per-level metrics | chain_comovement.json | dynamics.exact |
| Damped vs mirror | 7e-13 | solver_cross_agreement.json | solve.advanced |
| Gambit agreement 32×2 | 1.2e-9 | gambit_agreement.json | solve.advanced |
| Implicit-diff Jacobian ≡ χ^eq | machine eps | implicit_chi_agreement.json | solve.advanced |
| Branch through fixed-λ points | 8.6e-13 | branch_agreement.json | solve.branch |
| Beckmann ∇ ≡ route costs; Fisk KKT; DF; toll-χ FD | 0 / 7e-15 / 0 / 2.8e-9 | population_identities.json | population.core |
| Sioux Falls ℛ / DF / KKT; UE diagnostic 0.73 (diagnostic only) | 5.7e-17 / 0 / 7e-15 | sioux_falls_calibration.json | domains.congestion |
| Blotto α / ℛ / EPR / degenerate null | 0.69 / 0.12 / 0.098 / 5e-32 | blotto_readings.json | domains.blotto |
| Phase surface; wedge onset (median α=0.5, 0.2-fraction 0.4); criticality escape | full surface | phase_map.json + phase_map_surface.json | science.phase_map |
| Finding 3: H2 ρ(ℛ,1/den)=0.993; H1 refuted ρ(num,EPR)=−0.37 [CI excl. 0] | per-level metrics + CI | decoupling_mechanism.json | science.decoupling |

Findings log: memory/findings.md F-0001..F-0007 (incl. the refuted hypothesis and its post-hoc provenance).
Adversarial record: gates/*.yaml objections blocks; every unit red-team-signed.
