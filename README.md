# SAGE — Strategic Agent Game Engine

> SAGE — a computational framework for stochastic strategic interaction: Nash and quantal-response equilibria, potential and non-potential games, entropy-regularised response, and non-equilibrium strategic dynamics, with empirical estimation across pricing, energy, congestion and allocation domains.

| Level | Name | Where |
|---|---|---|
| Umbrella project / monorepo | **SAGE** | this repo |
| Python library (PyPI + import) | **`strataq`** | [`packages/strataq`](packages/strataq) |
| Research sub-framework | **ThermoQRE** | [`packages/strataq/thermo`](packages/strataq/thermo), [`papers/`](papers), [`research/`](research) |
| Reference app | **SAGE Labs** | [`apps/web`](apps/web) |

The package is `strataq`, not `sage` — SageMath owns the `sage` import. It is always `import strataq`.

## What this is

A set of **measuring instruments for strategic systems** — a susceptibility meter (`chi_equilibrium`), a reciprocity meter (`reciprocity_defect`), an entropy-production meter, and a phase locator — built on JAX, pointed at systems where the answer is known (congestion games, Colonel Blotto), where it matters commercially (retail pricing, electricity bidding), and where nobody has looked.

Three engines, strictly separated:

- **`finite/`** — finite N-player strategic form: the resolvent `(I − SB)⁻¹`, Hodge decomposition on Cartesian-product graphs, harmonic fraction α.
- **`population/`** — population/aggregative games (Sandholm): payoff field `F(x)`, Fisk stochastic user equilibrium, Beckmann potential.
- **`bayesian/`** — incomplete information. Deferred by explicit decision.

Domains (pricing, congestion, Blotto, electricity, security, sports) are **plugins**: five declared objects, zero core changes, and a mandatory `ConjugateFieldSpec` naming the observable payoff perturbation — or declaring there isn't one, in which case the response instruments refuse to run.

## Orientation

- The science and its confidence-labelled claim ledger: [`research/THERMOQRE_PROGRAMME_v3.md`](research/THERMOQRE_PROGRAMME_v3.md) and [`memory/claims.md`](memory/claims.md)
- The multi-domain expansion and the conjugate-field criterion: [`research/THERMOQRE_DOMAINS_v1.md`](research/THERMOQRE_DOMAINS_v1.md)
- How work gets closed (gates, adversarial review): [`gates/`](gates) and [`CLAUDE.md`](CLAUDE.md)
- Progress dashboard: `/progress` on the project Pages site, regenerated on every merge to `main`.

## Licence

Library code: Apache-2.0. App/API: source-available (intent recorded in the research docs). Dominick's-derived data artefacts: CC-BY-NC-4.0, everywhere, always.
