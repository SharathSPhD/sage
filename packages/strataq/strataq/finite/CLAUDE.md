# finite/ — Engine 1 rules (finite N-player strategic form)

- **Tangent space, always.** Cᵢ is rank-deficient by construction. All linear algebra on (I − SB) happens on the mean-zero subspace T via an explicit orthonormal basis (Helmert, or QR of I − (1/m)𝟙𝟙ᵀ). A bug here silently produces a spurious zero eigenvalue and a **false criticality reading**. The projection has its own dedicated tests.
- **Normalised game, always.** Decomposition runs on the effective (strategically equivalent, normalised) game, never the raw payoff tensor — full externality symmetry is sufficient but not necessary for potentiality (PROGRAMME v3 §1.1, arXiv:2405.07224 Lemma C.2). Test: strategically equivalent games give identical α.
- **Payoff scale folding.** Rescale payoffs to unit range internally, fold the scale into λ, report both `lam` and `lambda_normalised`.
- **α comes from the intrinsic Candogan flow decomposition** via the separable Kronecker transform (PROGRAMME v3 §1.2) — never by perturbing a Jacobian; the skew part of a Jacobian is coordinate-dependent and generally not integrable.
- **Matrix-free conventions**: Bv is a vmapped contraction; GMRES away from criticality, dense tangent-space eigendecomposition near it, switched on `distance_to_criticality` (config: `engines/finite.yaml`).
- Expose `distance_to_criticality = 1 − ρ(SB)`; warn below the configured threshold instead of returning a huge χ.
- `log_softmax`/`logsumexp` only.
