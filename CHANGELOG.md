# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: SemVer; stage completions are tagged `v0.<stage>.0`.

## [Unreleased]

### Added
- Advanced solvers (unit: solve.advanced): magnetic mirror descent (last-iterate, Sokota et al. 2023), strategy dispatch, implicit differentiation through the QRE fixed point via custom VJP sharing the (I-SB) resolvent; pygambit cross-validation (fixed-lambda agreement 1e-8 on 32 games x 2 lambdas).
- Exact non-equilibrium layer (unit: dynamics.exact): Glauber-logit generator on the joint profile space, stationary distribution, probability currents J*, exact entropy production rate, thermo_read facade. Calibration: pi = e^(lambda Phi)/Z to 1e-10 on congestion (K3), EPR/J* < 1e-12 on potential games, NESS on RPS/matching-pennies; first C1 co-movement data rho(EPR, alpha) = 0.990, rho(EPR, R) = 0.993 (n = 1000).
- Engine 1 core + first instruments (units: finite.core, finite.decompose.hodge, finite.response.reciprocity): dense tensor games, normalisation, verified game library (Rosenthal congestion with explicit potential, coordination, common-interest, RPS family, matching pennies); JIT damped logit-QRE solver; tangent-space machinery (Helmert basis); chi_partial, chi_equilibrium (Result 1), reciprocity_defect (Result 2), strategic_spectrum + critical_lambda; separable Kronecker subset transform, m-weighted Candogan Hodge projection, alpha(), make_family() at exact target alpha. Calibration: R <= 9e-17 on potential games, R >= 0.43 on harmonic games, Spearman rho(R, alpha) = 0.982 (n = 2000), chi-FD agreement 1.3e-8.
- Stage 0 foundation: monorepo scaffold, uv workspace (`strataq`, `strataq-client`, `strataq-bench`), plugin protocols (`PayoffOracle`, `ActionGridBuilder`, `ConjugateFieldSpec`, `DatasetLoader`, `LearnPageSpec`), config tree (Hydra + Pydantic), CLAUDE.md hierarchy, 13 subagent definitions, skills (gate-runner, triz-engine, adversarial-review, experiment-runner, docs-sync, release), enforcement hooks, gates machinery, seeded memory (claims ledger, literature, ADRs, glossary), CI, docs skeleton, progress dashboard. (unit: stage0.foundation)
