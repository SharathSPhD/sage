# Domain-adapted contradiction matrix

The classical 39×39 matrix is available through the `triz-knowledge` MCP server (`lookup_matrix`, `suggest_parameters`). This file is the **project-adapted** reduction: the parameters that actually recur in SAGE, and the principle shortlists that have earned their place. Use this first; fall back to the classical matrix when the conflict doesn't fit these parameters.

## Project parameters

| # | Parameter | Typical readings |
|---|---|---|
| P1 | Numerical exactness | identity tolerance, decomposition orthogonality |
| P2 | Scale / problem size | N, m, tensor entries, network size |
| P3 | Speed / latency | solve time, API latency, CI wall-clock |
| P4 | Conditioning / stability | distance to criticality, condition numbers |
| P5 | Generality | engines × domains covered by one code path |
| P6 | Domain validity | claims checkable in a specific domain |
| P7 | Statistical power | n, CI width, FDR budget |
| P8 | Honesty / claim strength | tier level, referee-robustness |
| P9 | Reproducibility | seed discipline, artifact regenerability |
| P10 | Development velocity | units closed per week |

## Earned shortlists (improving row degrades column → try principles)

| Improving ↓ / Degrading → | P2 scale | P3 speed | P4 conditioning | P6 domain validity | P10 velocity |
|---|---|---|---|---|---|
| **P1 exactness** | 1, 17, 28 (separable/spectral structure) | 21, 10 (bracket coarse, polish exact) | 9, 35 (project first, rescale) | 26, 27 (small exact cross-checks) | 11, 26 (oracle fallbacks) |
| **P3 speed** | 28, 5 (matrix-free, shared resolvent) | — | **15, 3 (switch methods by region)** | 24 (intermediary oracle) | 10 (warm-up, caching) |
| **P5 generality** | 6, 24 (universal oracle protocol) | 30 (thin adapters) | 3 (per-engine operators) | **23, 3 (declared capability: ConjugateFieldSpec)** | 1 (plugin segmentation) |
| **P7 power** | 16 (bounds not points) | 19, 25 (scheduled, self-documenting runs) | 26 (synthetic ground truth) | 33 (same machinery across domains) | 23 (gates force n-justification early) |
| **P8 honesty** | — | — | 32 (visible tiers/warnings) | 38 (adversarial intensification) | 34, 23 (supersede claims, feedback) |
| **P9 reproducibility** | 39 (frozen configs, hermetic CI) | 27 (fast subset) | 26 (golden copies) | 25 (resolved-config-beside-results) | 10 (pre-commit before CI) |

Bold cells are the two contradictions this project has already resolved structurally; their ADRs are the template.

## Physical contradictions — go to separation first

One parameter needing opposite values (exact AND never materialised; general AND domain-specific; fast AND well-conditioned) is resolved by separation (time / space / condition / scale) **before** consulting any matrix. See SKILL.md step 3.
