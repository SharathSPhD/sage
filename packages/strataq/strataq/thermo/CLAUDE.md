# thermo/ — ThermoQRE rules

- **Confidence tiering is mandatory.** Every public function's docstring `References` section states which tier its claim sits in, per `memory/claims.md`:
  - `exact` — proved identities (Gibbs variational principle, CGF, Glauber/Gibbs in potential games, exact EPR formula). Implement, cite, never claim as new.
  - `derived` — Results 1–2 (resolvent, reciprocity transfer) and anything proved in-house and checked numerically.
  - `conjectured` — e.g. the cycling → dissipation chain tightness. State what would count as wrong.
  - `speculative` — exploratory framings. Keep out of user-facing docs except clearly marked.
- The physics is **exact only where it is exact**: Gibbs measure statements hold for potential games; for non-potential games what survives is the NESS framing, currents, EPR, TUR. Never let "temperature" language imply a global potential exists where it doesn't.
- EPR estimators (KLD k-th order, TUR bound, NEEP-style) are validated on synthetic trajectories with known ground truth **before** touching real data. TUR lower bound is the headline empirical number — it degrades gracefully under partial observation.
- Entropy production is computed from the Glauber generator on the joint profile space (`core/dynamics`); this module owns the thermodynamic interpretation layer (Hatano–Sasa split, TUR, Jarzynski/Crooks for λ-quenches), not the Markov machinery itself.
