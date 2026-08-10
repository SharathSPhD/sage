# Glossary

Definitions the dashboard's hover-defs and the docs draw from. One line each; precision over brevity where they conflict.

- **λ (lambda)** — logit precision: how strongly payoff differences translate into choice probabilities. λ=0 uniform, λ→∞ best response. Inverse temperature in the thermodynamic reading; inverse information price in the rational-inattention reading. Not scale-free: report both `lam` and `lambda_normalised = lam × payoff_range`.
- **α (alpha)** — harmonic fraction of the **normalised** game: ‖u^H‖/(‖u^P‖+‖u^H‖) ∈ [0,1] from the Candogan flow decomposition. α=0 potential (congestion), α≈1 harmonic (RPS/Blotto-like). The programme's x-axis.
- **ℛ (reciprocity defect)** — ‖χ^eq − χ^eqᵀ‖_F / ‖χ^eq + χ^eqᵀ‖_F. Reads 0 exactly when the normalised game is potential (Result 2); λ-free; estimable from pass-through asymmetry.
- **χ (susceptibility)** — response of equilibrium play to a payoff perturbation h. `chi_partial` = λC (opponents frozen, exact FDT); `chi_equilibrium` = (I − SB)⁻¹S (full strategic feedback, Result 1).
- **S** — blockdiag(λᵢCᵢ): the partial-susceptibility block matrix; symmetric PSD, positive definite on the tangent space.
- **B** — cross-payoff operator: B_ij(a,b) = ∂U_i(a)/∂σ_j(b), zero diagonal blocks. Its symmetry (on T, normalised game) is potentiality.
- **C** — per-player choice covariance diag(σ) − σσᵀ; rank-deficient by construction (hence the tangent space).
- **Tangent space (T)** — the mean-zero subspace ⊕ᵢ{v : 𝟙ᵀv = 0} where all (I − SB) algebra must live; skipping the projection fakes criticality.
- **Normalised (effective) game** — the strategically-equivalent representative on which decomposition must run; full externality symmetry is sufficient but not necessary on the raw game.
- **distance_to_criticality** — 1 − ρ(SB). Zero = a bifurcation of the QRE correspondence; the phase locator's needle.
- **EPR (σ_EP)** — entropy production rate of the Glauber chain at stationarity; 0 iff detailed balance (potential game); the dissipation meter.
- **J*** — stationary probability current on the profile space; nonzero = NESS; circulation = cycling.
- **TUR** — thermodynamic uncertainty relation: a lower bound on dissipation needed to sustain a current of given regularity; the headline empirical estimator (degrades gracefully under partial observation). The sample version (`tur_epr_bound`) is a point estimate that can straddle the true EPR near saturation; the certifiable statement is the lower bootstrap quantile (`tur_epr_bound_ci`).
- **KLD estimator** — plug-in k-th-order-Markov estimate of irreversibility from a trajectory: (1/kτ)·KL(forward blocks ‖ reversed blocks). For a stationary Markov chain the (k+1)-block KLD equals k × per-step EP exactly; needs n ≫ n_states^(k+1) samples, and data-starved k underestimates.
- **Uniformisation** — exact simulation of a CTMC as its skeleton chain P = I + L/Λ with Exponential(Λ) holding times; skeleton per-step entropy production = EPR/Λ.
- **NESS** — non-equilibrium steady state: stationary but current-carrying; where non-potential strategic systems live.
- **SUE** — stochastic user equilibrium (Fisk 1980): logit route choice; entropy-regularised Wardrop with a known convex potential; the α=0 calibration anchor.
- **Conjugate field (h)** — the observable payoff perturbation a domain declares via `ConjugateFieldSpec` (tolls, budgets, fuel shocks). No field ⟹ no response instruments for that domain.
- **Engine** — a mathematical setting with its own response operators and decomposition (`finite`, `population`, `bayesian`). New engines require an ADR.
- **Plugin** — a domain: exactly {oracle, grid, field, loader, learn} on an existing engine, zero core changes.
- **Gate** — the closure contract for a work unit: code + domain + statistical + documentation + adversarial sections, all green, artifacts regenerable. Closes on domain validation, not green tests.
- **QRE** — quantal response equilibrium: fixed point of logit responses to logit responses (McKelvey–Palfrey 1995). The equilibrium object is a distribution, not a point.
- **Glauber dynamics** — single-site logit revision Markov chain on the joint profile space; reversible iff the game is potential.
- **Hodge decomposition** — orthogonal split of a game into potential ⊕ harmonic ⊕ nonstrategic (Candogan et al. 2011); computed here by the separable Kronecker transform.
