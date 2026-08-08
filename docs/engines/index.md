# Engines

Three mathematical engines, strictly separated (see the root CLAUDE.md and DOMAINS v1 §3):

- **finite** — finite N-player strategic form: S, B, the resolvent (I − SB)⁻¹, Hodge decomposition via the separable Kronecker transform. Domains: pricing, electricity, Blotto, security, sports.
- **population** — continuum of agents (Sandholm): payoff field F(x), potentiality = symmetry of DF, logit equilibrium = Fisk's stochastic user equilibrium with the Beckmann potential. Domain: congestion — the programme's calibration standard.
- **bayesian** — incomplete information. Deferred by recorded decision (ADR-0004); required for auctions.

Solvers, implicit differentiation, dynamics and entropy machinery are shared in `core/`; only response operators and decomposition differ by engine.
