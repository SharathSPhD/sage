# SAGE

> A computational framework for stochastic strategic interaction: Nash and quantal-response equilibria, potential and non-potential games, entropy-regularised response, and non-equilibrium strategic dynamics, with empirical estimation across pricing, energy, congestion and allocation domains.

**The library is `strataq`** (`import strataq`, never `import sage` — SageMath owns that name).

SAGE builds **measuring instruments for strategic systems** and points them at things:

| Instrument | Reads | Status |
|---|---|---|
| `chi_equilibrium` | how equilibrium play responds to a payoff perturbation — the strategic resolvent (I − SB)⁻¹S | Stage 1 |
| `reciprocity_defect` (ℛ) | asymmetry of that response: exactly 0 on potential games, rising with harmonic content; λ-free | Stage 1 |
| `strategic_spectrum` | eigenvalues of SB: distance to criticality, bifurcation type | Stage 1 |
| `entropy_production_rate` | dissipation of the strategic dynamics: 0 iff detailed balance | Stage 1 |
| `alpha` | harmonic fraction of the (normalised) game via the separable Hodge transform | Stage 1 |

Where they get pointed: systems where the answer is **known** (congestion networks — exact potential, α = 0; Colonel Blotto — strongly non-potential), where it **matters commercially** (retail pricing, electricity bidding), and where nobody has looked.

- **[Progress dashboard](progress/index.md)** — what works now, gate states, claim ledger.
- **[Concepts](concepts/tiers.md)** — the confidence tiers every claim carries.
- **[Architecture patterns](architecture/patterns.md)** — how the code is built.

Pre-alpha; the library reaches PyPI when Stage 1 gates are green.
