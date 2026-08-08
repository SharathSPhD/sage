# The reciprocity meter — ℛ

**What it reads.** How asymmetric the equilibrium give-and-take between players is:

$$\mathcal{R} \;=\; \frac{\lVert \chi^{\mathrm{eq}} - (\chi^{\mathrm{eq}})^\top\rVert_F}{\lVert \chi^{\mathrm{eq}} + (\chi^{\mathrm{eq}})^\top\rVert_F} \;\in\; [0, \infty),$$

zero exactly when the response is fully reciprocal; values above 1 are possible and meaningful — they say the antisymmetric (circulating) part of the response *dominates* the symmetric part, as it does in matching pennies (ℛ = 1.2 at λ = 1.2),

where $\chi^{\mathrm{eq}} = (I - SB)^{-1}S$ is the equilibrium susceptibility — the response of everyone's play to a small payoff perturbation, *after* all strategic feedback settles.

**Why it means something** *(tier: derived — see the [claims ledger](https://github.com/SharathSPhD/sage/blob/main/memory/claims.md), entries R1/N1/N2)*. The reciprocity transfer (Result 2): $\chi^{\mathrm{eq}}$ is symmetric **iff** $S(B-B^\top)S = 0$, i.e. iff the normalised game has zero harmonic component. Strategic feedback neither creates nor destroys reciprocity — so the observable response inherits the symmetry of the unobservable payoff operator exactly. Consequences:

- ℛ = 0 exactly on potential games (where QRE is a Gibbs equilibrium, Onsager-style reciprocity holds);
- ℛ grows with harmonic content. **The λ-free property is the symmetry statement, not the magnitude**: whether ℛ = 0 (reciprocity holds ⟺ potential) does not depend on λ, which is what answers the Haile–Hortaçsu–Kosenok critique — there is no noise parameter that can absorb an asymmetry. The *magnitude* of ℛ does vary with λ (approximately ∝ λ at small λ; findings F-0002), so cross-system comparisons of ℛ levels must hold λ fixed or report it;
- operationally, ℛ is estimable from *cross-agent pass-through asymmetry*: how much player $i$ moves when $j$'s costs move, versus the reverse — no payoff knowledge required.

**How it is computed.** All algebra on the tangent space via an explicit Helmert basis (a rank-deficiency slip here fakes criticality); the resolvent is shared with implicit differentiation and the spectral phase locator. See `strataq.finite.response.reciprocity`.

**Calibration state.** Gate `finite.response.reciprocity`:

| Reading | Requirement | Artifact |
|---|---|---|
| 5 exact potential games | ℛ < 10⁻¹⁰ | `reciprocity_potential.json` |
| RPS family + matching pennies | ℛ > 0.1 | `reciprocity_harmonic.json` |
| 2,000-game α sweep at fixed λ = 1.2 | Spearman ρ(ℛ, α) > 0.9, bootstrap CI | `reciprocity_alpha_sweep.json` |
| χ^eq vs finite differences | agreement to 10⁻⁶ on 50 games | `chi_fd_agreement.json` |

The sweep holds λ fixed, so λ cannot drive the correlation; ρ < 1 reflects genuine cross-game structure at equal α (e.g. RPS-3 and RPS-5 both sit at α = 1 but read different ℛ). Reported readings are rank-order evidence, not a functional law.

All artifacts regenerate from a fixed seed via `uv run python -m experiments.reciprocity_calibration` (`make reproduce`).

**Limitations, stated once.** ℛ is a property of the *normalised* game at a *specific* QRE point; near criticality (`distance_to_criticality` below the configured threshold) magnitudes of χ are unreliable and the API flags rather than reports. Empirical estimation from pass-through (Stage 3) inherits the identification caveats of the demand stage, not of ℛ itself.
