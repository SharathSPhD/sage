# Congestion — the α = 0 anchor and calibration standard

**Engine**: population (Engine 2). **Conjugate field**: link tolls — exactly linear, the cleanest field in the programme. **Loader**: TNTP (`github.com/bstabler/TransportationNetworks`), Sioux Falls first, fetched and cached at runtime.

**Why it is the calibration standard.** Every other domain estimates payoffs before the instruments can read anything; congestion doesn't. BPR costs are given, the potential is the Beckmann integral (∇Beckmann ≡ route costs — verified by autodiff to 10⁻¹²), and logit route choice *is* Fisk's stochastic user equilibrium. So the meters can be checked against known ground truth **on real network data**: if anything reads nonzero here, the bug is in the code, not the world.

**Measured** (gates `population.core` + `domains.congestion`, artifacts regenerable):

| Reading | Braess diamond (synthetic) | Sioux Falls (real, top-8 OD, k=3 routes) |
|---|---|---|
| ℛ (toll-response reciprocity) | < 10⁻¹⁰ | **5.7×10⁻¹⁷** |
| DF symmetry defect | < 10⁻¹⁴ | 0.0 |
| Fisk KKT spread | 7×10⁻¹⁵ | 7×10⁻¹⁵ |
| toll-χ vs finite differences | < 10⁻⁵ | — |

**Solver note.** The damped logit-assignment map is *not* a contraction on steep-BPR networks (it two-cycles on the Braess diamond); SUE is solved by damped Newton on the convex Fisk program in per-OD tangent coordinates — SPD Hessian, global convergence, machine-precision KKT.

**Scope, stated once.** Route sets are the k shortest by free-flow time and the calibration uses an OD subset: the instrument identities are exact *within* the route set regardless, but computed link flows are not comparable to the repo's best-known UE flows (which carry all 528 OD pairs) — the θ→∞ link-flow gap in the artifact is a diagnostic, never a certificate. Empirical claims about actual driver dispersion would need observed route shares, which TNTP does not provide (calibration ≠ empirics).
