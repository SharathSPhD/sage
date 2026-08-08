# population/ — Engine 2 rules (population / aggregative games)

- State is a **distribution over strategies**, not a profile of mixed strategies. Sandholm's setting.
- Payoff field F(x) with Jacobian DF(x). **Potentiality is exactness of F, tested by symmetry of DF** (Sandholm's externality symmetry — a known result; cite, never claim).
- Logit equilibrium here **is Fisk's (1980) stochastic user equilibrium**, with the known convex potential Σₐ∫₀^{xₐ}cₐ(u)du + λ⁻¹Σₐ xₐ log xₐ (Beckmann + entropy). This is the programme's only analytically-known potential with real data — it is the calibration standard; treat its tests as the regression anchor for every instrument.
- **The finite-game Hodge machinery does not transfer here.** No Kronecker transform, no Candogan flow decomposition on population states. Potentiality testing is DF symmetry, full stop. Do not import from `finite/decompose`.
- The analogue of B is DF; the analogue of S is λ·C(x) on the population state. Response operators are re-derived, not copied.
- λ = Fisk's dispersion parameter θ; 45 years of estimated θ values exist to sanity-check against.
