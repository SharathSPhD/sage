# ThermoQRE v2 — Research Programme and Build Specification

**Status:** Master context document. Supersedes `research1.rtf` and the v1 research report.
**Audience:** (a) the principal investigator, (b) a Claude Code agent building the software.
**Author context:** PI is a Python expert, game theorist, and non-equilibrium thermodynamicist. Prior work: DreamPrice (`github.com/SharathSPhD/dreamprice`, `hf.co/qbz506/dreamprice-cso`).
**Licence intent:** library Apache-2.0; app/API source-available; Dominick's-derived artefacts remain CC-BY-NC-4.0.

---

## 0. How to use this document

Sections 1–6 are **science**: what is being claimed, what is already known, what is new, what would falsify it. Sections 7–13 are **engineering**: the library, the API, the app, and the build order.

A Claude Code agent should treat §7–§13 as the specification and §1–§6 as the rationale that constrains design decisions. When an engineering choice is ambiguous, resolve it in favour of whatever makes the §3 theorems directly computable and directly visible to a user.

Two standing rules for the agent:

1. **Do not write half-files.** Either write a module fully or edit an existing one with targeted replacements. Never assume unseen code exists.
2. **Every theoretical object in §3 must be a first-class, named, tested function in the library.** If a quantity appears in a theorem, it appears in the API surface. This is the discipline that makes the app "more than Gambit."

---

## 1. What changed from v1, and why

The v1 report proposed: MaxEnt/QRE equivalence → fluctuation–dissipation estimator of λ → Dominick's empirics. A careful internal review raised four objections, all of which I accept, and which restructure the programme:

**Objection 1 — potential games as destination vs. laboratory.** v1 treated the potential-game/Gibbs correspondence as "the clean part" and non-potential games as a caveat. Correct restructuring: potential games are the *controlled laboratory*, near-potential games are the *experiment*, and generic oligopoly is the *target*. The scientific question is not "does QRE = Gibbs?" (it does, in potential games, trivially) but **"how does the equilibrium fluctuation–response relation deform as strategic detailed balance breaks?"**

**Objection 2 — the FDT claim was too strong.** v1 asserted that divergence between the dispersion estimator and MLE diagnoses broken detailed balance. It does not, because demand misspecification, measurement error, non-stationarity, endogeneity, λ-heterogeneity and promotions all produce the same divergence. This is now demoted to "evidence to investigate" in empirical data, and promoted to a *sharp prediction* in the synthetic laboratory where those confounds are absent by construction.

**Objection 3 — partial vs. equilibrium susceptibility were conflated.** The single-agent identity ∂E[a]/∂h = λ·Var(a) is exact but is a *partial* (opponents-frozen) response. The equilibrium response includes strategic feedback. v1 did not distinguish these. **Resolving this objection produced the flagship result of the programme** (§3.4): the two differ by a strategic feedback resolvent, and the equilibrium susceptibility is Onsager-reciprocal **iff** the game is potential.

**Objection 4 — this should be more than a paper.** Correct. The programme's output is a library, an API, and an app, of which the papers are a by-product. §7–§13.

**New information not available at v1:**

- The "dreamprice" dataset is the PI's own prior work: `qbz506/dreamprice-dominicks-cso` (Dominick's canned soup, 500k rows, 529 cols, CC-BY-NC-4.0), with an accompanying model `qbz506/dreamprice-cso` (DreamerV3-style RSSM, Mamba-2 backbone, ~22M params) and a Gradio demo space. **All Dominick's data engineering is already done.** The model card reports a DML-PLIV elasticity of −0.940 (SE 0.006, first-stage F = 23,381) frozen into a causal demand decoder, with a Hausman leave-one-out-mean instrument. This means the demand layer of ThermoQRE is *already built and identified*.
- Candogan, Menache, Ozdaglar & Parrilo, "Flows and Decompositions of Games: Harmonic and Potential Games," *Mathematics of Operations Research* 36(3):474–503 (2011), arXiv:1005.2405 — canonical direct-sum decomposition of any finite strategic-form game into **potential ⊕ harmonic ⊕ nonstrategic** components. This makes the reviewer's α-knob a canonical, measurable quantity rather than an arbitrary interpolation. Follow-ups: Candogan, Ozdaglar & Parrilo, "Dynamics in near-potential games," *GEB* 82:66–90 (2013); "A projection framework for near-potential games," CDC 2010.
- Kianercy & Galstyan, "Dynamics of Boltzmann Q-learning in two-player two-action games," *Phys. Rev. E* 85(4) (2012) — logit/Boltzmann learning dynamics with explicit temperature, the nearest existing physics treatment.
- Gambit is at 16.7.0 (2026) and `pygambit` now ships `logit_solve`, `logit_solve_branch`, `logit_solve_lambda`, and `logit_estimate` (MLE along the correspondence, added 16.3.0, with an `use_empirical` fast path). This defines precisely what "more than Gambit" must mean (§7.1).
- JAXopt is in maintenance mode and being merged into Optax. **Optimistix** (Kidger) is the current recommended JAX library for root-finding / fixed points / least squares, with **Lineax** for linear solves and **Equinox** for module structure.

---

## 2. The revised central question and hypotheses

**Central question.**
> How does stochastic strategic behaviour in discrete-price markets transition from equilibrium-like Gibbs behaviour to genuinely non-equilibrium pricing, and can the resulting fluctuation–response signatures be detected in real pricing data?

This is deliberately narrower than "can statistical mechanics explain pricing?" (vulnerable to the physics-metaphor charge) and broader than "can QRE improve pricing?" (too narrow to be a programme).

**H1 — Equilibrium benchmark (potential games).** In an exact potential game, logit QRE ≡ entropy-regularised Nash ≡ Gibbs measure over the potential; logit dynamics are reversible; stationary probability current J* = 0; entropy production σ = 0; equilibrium susceptibility is symmetric (Onsager reciprocal); and the static FDT χ = λ·Cov holds exactly for the partial response.

**H2 — Departure from potentiality.** As the harmonic component α grows (Candogan norm fraction): detailed balance breaks; J* ≠ 0; entropy production rises monotonically; the reciprocity defect of the equilibrium susceptibility rises **in exact proportion to the harmonic component** (§3.4); and beyond a critical (λ, α) locus, Hopf bifurcation produces sustained price cycles.

**H3 — Real pricing.** Observed price distributions in scanner data exhibit measurable departures from deterministic Nash. Conditional on an independently identified demand system (DreamPrice), those departures are decomposable into (i) a stochastic-response component (λ finite), (ii) a non-potential/strategic-circulation component (reciprocity defect > 0), and (iii) a residual attributable to misspecification. The programme's honest claim is about (i) and (ii) *given* a stated demand model, with (iii) bounded by sensitivity analysis.

**Explicitly not claimed.** That ThermoQRE beats a well-tuned demand-based optimiser at setting tomorrow's price. That would be a nice bonus (and the app measures it), but the programme does not depend on it.

---

## 3. Theory: what is exact, what is new

Notation. Players $i \in \{1,\dots,N\}$ (firms). Finite action sets $A_i$ (price grids), $|A_i| = m_i$. Mixed strategies $\sigma_i \in \Delta(A_i)$, full support assumed throughout. Expected payoff to pure action $a$:
$$U_i(a;\sigma_{-i}) = \sum_{a_{-i}} \Big(\prod_{j\ne i}\sigma_j(a_j)\Big)\, u_i(a, a_{-i}).$$
Logit QRE with precision $\lambda_i$:
$$\sigma_i(a) = \frac{\exp\{\lambda_i U_i(a;\sigma_{-i})\}}{\sum_{b\in A_i}\exp\{\lambda_i U_i(b;\sigma_{-i})\}}.$$

### 3.1 Tier-A results: exact equivalences (already in the literature — implement, do not claim)

**(A1) Entropy-regularised best response.** $\sigma_i = \arg\max_{\sigma\in\Delta(A_i)}\{\mathbb{E}_\sigma[U_i] + \lambda_i^{-1} H(\sigma)\}$, $H(\sigma) = -\sum_a \sigma(a)\log\sigma(a)$. Exact. (Gibbs variational principle; Fudenberg–Iijima–Strzalecki perturbed utility.)

**(A2) Log-partition function is the CGF.** With $\psi_i(\lambda U_i) = \log\sum_a e^{\lambda_i U_i(a)}$,
$$\nabla_{U}\psi_i = \lambda_i\,\sigma_i, \qquad \nabla^2_{U}\psi_i = \lambda_i^2\, C_i, \qquad C_i := \operatorname{diag}(\sigma_i) - \sigma_i\sigma_i^\top.$$
$C_i$ is the covariance of the action-indicator vector. Exact; this is the Williams–Daly–Zachary theorem plus exponential-family structure.

**(A3) Gibbs measure in potential games.** If $u_i(a_i',a_{-i}) - u_i(a_i,a_{-i}) = \Phi(a_i',a_{-i}) - \Phi(a_i,a_{-i})$ for all $i$, then asynchronous logit (Glauber) dynamics with common $\lambda$ are reversible with stationary distribution
$$\pi(a) = Z^{-1}\exp\{\lambda\Phi(a)\}.$$
Exact. Detailed balance holds; $J^* = 0$; entropy production $= 0$.

**(A4) Rational-inattention micro-foundation.** $\lambda$ is the inverse shadow price of Shannon information (Matějka–McKay 2015; Fosgerau–Melo–de Palma–Shum 2020). Use this to defend λ against the "garbage parameter" charge — but note it does not by itself deliver identification.

**Rule for the agent and for papers:** A1–A4 are *cited*, not claimed. The library implements them as verified primitives with exact-identity unit tests.

### 3.2 The canonical α: Hodge decomposition of the game

Candogan et al. (2011) give a direct-sum decomposition of the space of finite games:
$$\mathcal{G} = \mathcal{G}_{\text{potential}} \oplus \mathcal{G}_{\text{harmonic}} \oplus \mathcal{G}_{\text{nonstrategic}}.$$
Write the payoff tuple $u = u^{P} + u^{H} + u^{N}$. Nonstrategic components do not affect any player's incentives (they are "own-action-independent" terms) and therefore **do not affect the QRE at all** — an important sanity invariant to unit-test.

Define the **harmonic fraction**
$$\alpha(u) := \frac{\|u^{H}\|}{\|u^{P}\| + \|u^{H}\|} \in [0,1],$$
with the norm induced by the flow inner product of Candogan et al. This replaces the reviewer's ad hoc $U = U^{\text{pot}} + \alpha U^{\text{rot}}$ with a canonical, basis-independent, and *measurable-from-payoffs* quantity. Synthetic game families are then generated by projecting onto the two subspaces and re-mixing at a target α.

Implementation note: the decomposition reduces to solving least-squares problems over the game graph incidence structure; for $N$ players with $m$ actions each the flow space has dimension $O(N m^N)$, so exact decomposition is feasible for small $(N, m)$ and must be done via iterative/Krylov methods or restricted to pairwise-marginal projections for large price grids. Provide both an exact path (small games, dense) and an approximate path (large games, matrix-free).

### 3.3 Partial susceptibility (exact, and the reviewer's correct restriction)

Introduce an external field $h_i \in \mathbb{R}^{m_i}$ added to player $i$'s own payoff: $u_i \mapsto u_i + h_i(a_i)$. Economically, $h$ is a per-action cost shock — a wholesale price change, a tax, a promotion subsidy — which is exactly the kind of shock scanner data contains.

**Partial (opponents frozen):**
$$\chi^{\text{part}}_i := \frac{\partial \sigma_i}{\partial h_i}\bigg|_{\sigma_{-i}} = \lambda_i C_i.$$
For a scalar observable $p$ (the price attached to each action) with field conjugate $h = \eta\, p$:
$$\frac{\partial \mathbb{E}_{\sigma_i}[p]}{\partial \eta}\bigg|_{\sigma_{-i}} = \lambda_i \operatorname{Var}_{\sigma_i}(p).$$
Exact, and it is a genuine static fluctuation–dissipation relation. **But it is not the object an econometrician measures**, because in data the opponents also move.

### 3.4 FLAGSHIP RESULT — equilibrium susceptibility and the Onsager criterion

Let $B$ denote the strategic cross-payoff operator, blocks
$$B_{ij}(a,b) := \frac{\partial U_i(a;\sigma_{-i})}{\partial \sigma_j(b)}, \quad j \ne i, \qquad B_{ii} := 0.$$
Let $S := \operatorname{blockdiag}(\lambda_i C_i) = \chi^{\text{part}}$, symmetric positive semi-definite (positive definite on the tangent space $T = \bigoplus_i \{v: \mathbf{1}^\top v = 0\}$ when $\sigma$ has full support).

Total differentiation of the fixed point gives $d\sigma = S(dh + B\,d\sigma)$, hence

> **Proposition 1 (strategic resolvent).** Whenever $(I - SB)$ is invertible on $T$,
> $$\boxed{\ \chi^{\text{eq}} := \frac{d\sigma^*}{dh} = (I - SB)^{-1} S = (I - \chi^{\text{part}}B)^{-1}\chi^{\text{part}}.\ }$$

The partial response is the $B \to 0$ limit. The resolvent $(I-SB)^{-1}$ is the strategic amplification/attenuation factor, and it is where all the interesting physics lives.

> **Proposition 2 (Onsager reciprocity ⟺ potentiality).** $\chi^{\text{eq}}$ is symmetric if and only if $S(B - B^\top)S = 0$. With full support ($S \succ 0$ on $T$) this holds iff $B = B^\top$ on $T$, i.e. iff
> $$\frac{\partial U_i(a;\sigma_{-i})}{\partial \sigma_j(b)} = \frac{\partial U_j(b;\sigma_{-j})}{\partial \sigma_i(a)} \quad \forall i \ne j,\ a,b,$$
> which is precisely the integrability condition characterising **exact potential games** (equivalently: zero harmonic component in the Candogan decomposition).

*Proof sketch.* $\chi^{\text{eq}} = (I-SB)^{-1}S$ is symmetric iff $(I-SB)^{-1}S = S(I-B^\top S)^{-1}$; multiplying out gives $S - SB^\top S = S - SBS$, i.e. $S(B-B^\top)S = 0$. Positive definiteness of $S$ on $T$ removes the sandwich. □

Define the **reciprocity defect**
$$\mathcal{R} := \frac{\big\|\chi^{\text{eq}} - (\chi^{\text{eq}})^\top\big\|_F}{\big\|\chi^{\text{eq}} + (\chi^{\text{eq}})^\top\big\|_F} \in [0,\infty).$$

**Why this matters, in four registers.**

1. *Physics.* Onsager reciprocity is the canonical equilibrium signature; it fails in non-equilibrium steady states and under broken time-reversal symmetry. Here it fails **exactly** when strategic detailed balance fails. This is a substantive isomorphism, not decoration: the same algebraic condition ($B$ symmetric) simultaneously delivers the Gibbs measure, zero probability current, zero entropy production, and Onsager reciprocity.
2. *Game theory.* It gives an operational, response-based characterisation of potentiality that requires no knowledge of $\Phi$ — you can test for potentiality by perturbing and measuring, without ever constructing the potential.
3. *Econometrics.* $\mathcal{R}$ is estimable from **cross-firm cost-shock pass-through asymmetry**: how much does firm $i$'s price distribution move when firm $j$'s cost moves, versus the reverse. Asymmetric cross-response ⟹ harmonic strategic content. This is a new empirical object.
4. *Answers Objection 3 directly.* The reviewer asked how the partial and equilibrium responses differ. Proposition 1 answers it exactly, and Proposition 2 shows the difference is not a nuisance but the carrier of the scientific signal.

**Novelty assessment.** The strategic-resolvent form (Prop. 1) is essentially a comparative-statics calculation and is likely folklore in some form in the logit-equilibrium and mean-field-game literature; a literature check is required before claiming it (search terms: logit equilibrium comparative statics, quantal response comparative statics, aggregative games susceptibility, Brock–Durlauf multiplier). **Proposition 2 — the reciprocity ⟺ potentiality equivalence, and $\mathcal{R}$ as a harmonic-content measure — I could not locate in the literature and is the strongest candidate for a genuinely new theorem.** Verify against: Candogan et al. follow-ups, Sandholm's *Population Games and Evolutionary Dynamics*, Hofbauer–Sandholm on stable games, and the econophysics response-function literature (Garnier-Brun, Bouchaud & Benzaquen, *J. Phys. Complexity* 4:015004, 2023, which proves the consumer-side Slutsky-matrix fluctuation-response identity — the closest prior art, and the natural citation anchor).

### 3.5 The same operator gives the phase diagram

$(I - SB)$ is simultaneously:

- **the susceptibility resolvent** — $\chi^{\text{eq}}$ diverges as $(I-SB)$ approaches singularity, i.e. **critical opalescence**: susceptibility divergence at the transition;
- **the uniqueness/contraction certificate** — the logit best-response map is a contraction when the spectral radius $\rho(SB) < 1$; since $S \propto \lambda$, this yields an explicit small-λ uniqueness bound and a critical $\lambda_c$ where uniqueness can first fail;
- **the bifurcation detector on the QRE correspondence** — turning points on Gambit's principal branch are exactly where an eigenvalue of $SB$ crosses 1.

This yields a sharp typology to be mapped in $(\lambda, \alpha, N, \text{cross-elasticity})$ space:

| Spectral event | Interpretation | Observable signature |
|---|---|---|
| $\rho(SB) < 1$ | unique diffuse QRE | single branch, $\chi^{\text{eq}}$ bounded |
| real eigenvalue of $SB$ crosses $1$ | pitchfork/fold — multiplicity, coordination, symmetry breaking (Brock–Durlauf regime) | branch turning point, bimodal price distribution |
| complex pair of $SB$ crosses the unit circle (in the dynamics) | **Hopf bifurcation → sustained price cycles** | limit cycle, $J^*\ne0$, entropy production jumps, Edgeworth-like sawtooth |

**Prediction P1:** cycling requires non-potentiality. In an exact potential game $B = B^\top$, so $SB$ is similar to a symmetric matrix ($S^{1/2}BS^{1/2}$) and has **real spectrum** — no Hopf bifurcation is possible. Therefore *price cycles cannot arise from logit dynamics in an exact potential game*. This is a crisp, falsifiable, simulation-testable claim that ties the entire programme together: **cycling ⟹ non-potential ⟹ broken reciprocity ⟹ positive entropy production.**

### 3.6 Non-equilibrium layer

Construct the continuous-time logit (Glauber) Markov jump process on the joint action profile space with rates $w(a\to a')$ for single-player deviations. At stationarity $\pi^*$:
$$J^*(a,a') = \pi^*(a)w(a\to a') - \pi^*(a')w(a'\to a),$$
$$\sigma_{\text{EP}} = \tfrac{1}{2}\sum_{a,a'}\big[\pi^*(a)w(a\to a') - \pi^*(a')w(a'\to a)\big]\log\frac{\pi^*(a)w(a\to a')}{\pi^*(a')w(a'\to a)} \ \ge 0,$$
with equality iff detailed balance. Additional machinery to implement: Hatano–Sasa housekeeping/excess decomposition; the thermodynamic uncertainty relation as a lower bound on the dissipation required to sustain a price cycle of given regularity; Crooks/Jarzynski for λ-quench protocols (annealing a market from diffuse to sharp pricing).

**For real data**, exact rates are unavailable. Implement three estimators of irreversibility from observed price-profile trajectories, in increasing order of assumption:
1. **KLD / $k$-th order Markov estimator** — $\hat\sigma^{(k)} = \frac{1}{k\tau}\sum P(Y_{0:k})\log\frac{P(Y_{0:k})}{P(Y_{k:0})}$ (time-reversal asymmetry of the observed symbol sequence). Assumption-light, data-hungry.
2. **TUR-based lower bound** — from the first two cumulants of an empirical current; gives a certified lower bound on dissipation.
3. **Neural estimator (NEEP-style)** — variational, best in high dimension, requires care about overfitting.
Cross-validate all three on synthetic data with known $\sigma_{\text{EP}}$ before touching Dominick's. **Report the TUR lower bound as the headline empirical number**, because it is a bound rather than a point estimate and is therefore honest under partial observation.

### 3.7 λ as a structured object, not a scalar nuisance

Replace scalar λ with $\lambda_{i,t} = \exp(x_{i,t}^\top\beta)$ where $x$ includes price dispersion, competitor-price volatility, time since last price change, promotion state, and information-quality proxies. Then:
- the RI micro-foundation (A4) interprets $\beta$ as sensitivity of the information price to the environment;
- menu-cost/adjustment-cost stories predict $\lambda$ falls with adjustment frictions;
- heterogeneity across stores is handled by hierarchical partial pooling.
The decomposition of λ into information cost, menu cost, forecast error, and organisational heterogeneity is **not** separately identified without exclusion restrictions. State this. Propose candidate exclusions (e.g. store-level staffing/turnover as an organisational shifter orthogonal to demand) and be explicit that they are assumptions.

---

## 4. Confronting the two hardest objections

### 4.1 Haile–Hortaçsu–Kosenok (AER 2008)

Unrestricted QRE has no empirical content in normal-form games: any behaviour is rationalisable. Three responses, in ascending strength, all of which the programme uses:

1. **Restrict to logit + regular QRE axioms** (Goeree–Holt–Palfrey). Necessary but weak on its own.
2. **Externally identify payoffs.** Once $u_i$ is pinned down by an independently estimated demand system (DreamPrice's IV-identified elasticity), the game is fixed and QRE becomes a one-parameter (or $\dim\beta$-parameter) restriction on the joint distribution of prices. Falsifiable. **This is the load-bearing response.**
3. **Test a response-based restriction that does not depend on λ at all.** Proposition 2 gives a restriction on the *symmetry* of the cross-response matrix that is independent of λ's value. That is a QRE-family prediction with no free parameter to absorb misfit — the strongest available answer to HHK, and worth foregrounding in the paper.

### 4.2 λ vs. collusion

Low λ and tacit collusion (Calvano et al., AER 2020) both raise prices and dispersion. Separators to implement and test:
- **Sign of the effect on the mean.** Pure logit noise around a Bertrand best response is approximately mean-preserving at leading order; collusion shifts the mean up systematically. Test the joint (mean, dispersion) signature, not either alone.
- **Reciprocity defect.** Collusive conduct implies coordinated, largely symmetric mutual responsiveness; harmonic circulation implies asymmetry. These make different predictions for $\mathcal{R}$.
- **Entropy production.** Collusive supergame punishment phases are transient excursions, not stationary circulation; cycling has persistent $J^*$.
- **Rivers–Vuong model selection** across the conduct menu {Bertrand, λ-QRE, collusive, QRE-with-collusive-conduct}, following Duarte–Magnolfi–Sølvsten–Sullivan (*Quantitative Economics* 15(3):571–606, 2024) with their weak-instrument diagnostic.
Be honest: this is **partial** separation, and it is the highest-risk claim in the programme.

---

## 5. Staged research plan with falsification thresholds

**Stage 0 — Literature verification (2 weeks).** Before writing a line of solver code, verify the novelty of Propositions 1 and 2 against the search terms in §3.4. If Prop. 2 exists, the programme survives but the flagship claim is downgraded to "we operationalise and estimate a known equivalence"; adjust paper framing, not the software.

**Stage 1 — Synthetic laboratory (months 0–4).**
Build the library core. Verify on exact potential games: QRE = Gibbs to machine precision; $J^*=0$; $\sigma_{\text{EP}}=0$; $\mathcal{R}=0$; $\chi^{\text{part}} = \lambda C$ exactly; spectrum of $SB$ real.
*Kill criterion:* if $\mathcal{R} \ne 0$ (beyond numerical tolerance) in a verified exact potential game with full support, the implementation or Prop. 2 is wrong. Stop and resolve.

**Stage 2 — The α-sweep experiment (months 4–8).**
Sweep $\alpha \in [0,1]$ and $\lambda \in [\lambda_{\min},\lambda_{\max}]$ over families of pricing games generated by Hodge re-mixing. Measure $\mathcal{R}$, $\|J^*\|$, $\sigma_{\text{EP}}$, cycle amplitude, spectral radius of $SB$, and the correspondence branch structure. Produce the phase diagram.
*Predictions:* $\mathcal{R}$ monotone increasing in α and zero at α=0; no Hopf at α=0 (P1); entropy production monotone in α; susceptibility diverges at the branch turning point.
*Kill criterion:* a Hopf bifurcation / sustained cycle at α=0 falsifies P1 and the reciprocity–cycling chain.

**Stage 3 — Dominick's empirics via DreamPrice (months 8–16).**
Use DreamPrice as the payoff oracle. Define markets (store × week × category), price grids from the empirical support, competitor sets from the price-zone and store-competition structure. Estimate λ three ways (MLE/NFXP, CCP two-step, dispersion/FDT), estimate $\mathcal{R}$ from cross-store cost-shock pass-through asymmetry, and estimate the TUR bound on dissipation.
*Honesty constraints, non-negotiable:* the rival-firm interpretation is weak (single chain, zone pricing, not true oligopoly); the AAC cost basis is economically wrong for margin; the data is 1989–97. Every headline number gets a sensitivity band across demand specifications.
*Kill criterion:* if the three λ estimates disagree beyond CI and the disagreement is not traceable to a diagnosable confound, the empirical claim is not made. Publish the synthetic results and the negative empirical finding.

**Stage 4 — Product (continuous, from month 2).**
Library → API → app, built incrementally alongside the science so that each theoretical object ships as a feature the moment it is verified. See §7–§13.

---

## 6. Publication plan (by-product, not goal)

- **Paper 1 (theory).** "Onsager Reciprocity and Potentiality in Quantal Response Equilibria." Props 1–2, the spectral typology, P1. Target: *GEB* / *Journal of Economic Theory* / *Physical Review E*.
- **Paper 2 (non-equilibrium).** "From Gibbs Pricing to Non-Equilibrium Pricing." The α-sweep, entropy production, Hopf/cycling, phase diagram. Target: *PRE* / *Journal of Economic Dynamics and Control*.
- **Paper 3 (empirical).** "Do Firms Price as Nash Optimisers?" Dominick's + DreamPrice, λ estimation, conduct testing. Target: *RAND* / *Quantitative Economics*.
- **Paper 4 (software).** JOSS / *Journal of Statistical Software* note on `thermoqre`.

---

# ENGINEERING SPECIFICATION

---

## 7. Product architecture

Three artefacts, strictly layered. Each layer depends only on the one below.

```
┌──────────────────────────────────────────────────────┐
│  APP        Next.js on Vercel                        │
│             Learn / Lab / Analyze                    │
└──────────────────────┬───────────────────────────────┘
                       │ HTTPS + JSON
┌──────────────────────▼───────────────────────────────┐
│  API        FastAPI on Render                        │
│             sync endpoints + async job queue         │
└──────────────────────┬───────────────────────────────┘
                       │ Python import
┌──────────────────────▼───────────────────────────────┐
│  LIBRARY    thermoqre  (JAX)          pip install    │
│             games · decompose · solve · response     │
│             dynamics · estimate · demand · data      │
└──────────────────────┬───────────────────────────────┘
                       │ adapter protocol
┌──────────────────────▼───────────────────────────────┐
│  ORACLES    DreamPrice · pyblp · log-log · custom    │
└──────────────────────────────────────────────────────┘
```

### 7.1 What makes this "more than Gambit"

Gambit is excellent and must be used as the ground-truth oracle. Be explicit about the delta, in the README and in the app:

| Capability | Gambit / pygambit | thermoqre |
|---|---|---|
| Logit QRE correspondence, small finite games | ✅ mature, authoritative | ✅ (validated *against* Gambit) |
| MLE of λ from play frequencies | ✅ `logit_estimate` | ✅ plus CCP, FDT, hierarchical, covariate-λ |
| Payoffs supplied as | a matrix you build | **a demand model + price grid** (PayoffOracle) |
| Large discrete price grids ($m \sim 10^2$, $N\sim 2$–$5$) | impractical | ✅ JAX, matrix-free, vectorised |
| Differentiable through the equilibrium | ✗ | ✅ implicit diff (Optimistix/Lineax) |
| Hodge decomposition, α measurement | ✗ | ✅ |
| Equilibrium susceptibility $\chi^{\text{eq}}$, reciprocity defect | ✗ | ✅ core |
| Probability currents, entropy production, TUR | ✗ | ✅ core |
| Bifurcation/Hopf classification of the correspondence | partial (turning points) | ✅ typed spectral classification |
| Panel-data pipeline, market definition, grid construction | ✗ | ✅ |
| Hosted API + interactive app + teaching mode | ✗ | ✅ |

**Design rule:** wherever Gambit can do it for small games, `thermoqre` must reproduce Gambit's answer to $10^{-8}$ in a CI test. Do not compete with Gambit on correctness; inherit it.

---

## 8. Library: `thermoqre`

### 8.1 Stack

- **Python** ≥ 3.11. **JAX** (CPU + CUDA extras). float64 enabled by default (`jax.config.update("jax_enable_x64", True)`) — the susceptibility and entropy-production computations are ill-conditioned near criticality and float32 is not acceptable.
- **Equinox** for module/pytree structure; **Optimistix** for root-find/fixed-point with implicit diff; **Lineax** for linear solves (including matrix-free CG/GMRES on $(I-SB)$); **Optax** for first-order optimisers.
- **NumPyro** for Bayesian hierarchical λ. **pygambit** (optional extra) for validation. **pyblp** (optional extra) for the BLP oracle. **polars** for panel data. **datasets**/`huggingface_hub` for loaders.
- Packaging: `pyproject.toml`, hatchling. Extras: `[gambit]`, `[blp]`, `[dreamprice]`, `[bayes]`, `[viz]`, `[all]`.
- Testing: pytest + hypothesis (property tests on the exact identities). CI: GitHub Actions, matrix over 3.11/3.12/3.13.

### 8.2 Module map

```
thermoqre/
  __init__.py
  types.py             # Game, QREPoint, Branch, Diagnostics dataclasses (equinox.Module)
  games/
    base.py            # Game protocol: payoff(profile) -> array; action grids; N; m
    tensor.py          # DenseTensorGame (small, exact)
    factored.py        # PricingGame: payoffs from a PayoffOracle over a price grid
    library.py         # canonical test games: matching pennies, coordination,
                       # congestion (exact potential), rock-paper-scissors (harmonic),
                       # Bertrand-logit duopoly, capacity-constrained pricing
  decompose/
    hodge.py           # potential / harmonic / nonstrategic projection; alpha()
    generate.py        # synthesise game families at target alpha
    potentialize.py    # nearest potential game (Candogan projection framework)
  solve/
    fixedpoint.py      # damped iteration, Anderson acceleration
    mirror.py          # magnetic mirror descent / entropy-regularised MD (last-iterate)
    homotopy.py        # predictor-corrector arclength continuation of the branch
    implicit.py        # implicit-diff wrapper: d(sigma*)/d(theta) for any theta
    validate.py        # cross-check against pygambit
  response/
    susceptibility.py  # chi_partial, chi_equilibrium (Prop 1), matrix-free option
    reciprocity.py     # reciprocity_defect (Prop 2), harmonic-content regression
    spectral.py        # spectrum of SB, rho(SB), critical lambda, bifurcation typing
    fdt.py             # FDT identity checks, dispersion-response estimator
  dynamics/
    markov.py          # Glauber/logit jump process generator on profile space
    stationary.py      # pi* via sparse eigensolve or power iteration
    currents.py        # J*, cycle decomposition
    entropy.py         # exact EPR; Hatano-Sasa; TUR bound
    estimators.py      # KLD k-th order, TUR-from-data, NEEP-style neural estimator
    simulate.py        # trajectory sampling; quench/anneal protocols (Jarzynski/Crooks)
  demand/
    oracle.py          # PayoffOracle protocol  <-- the key abstraction
    loglog.py          # constant-elasticity baseline
    logit_demand.py    # nested/multinomial logit demand
    blp.py             # pyblp adapter
    dreamprice.py      # DreamPrice world-model adapter
  estimate/
    mle.py             # NFXP: solve QRE inside the likelihood
    ccp.py             # Hotz-Miller two-step; avoids solving the equilibrium
    dispersion.py      # FDT/dispersion estimator of lambda
    hierarchical.py    # NumPyro partial pooling over stores/categories
    covariate.py       # lambda_{i,t} = exp(x'beta)
    compare.py         # Nash / QRE / level-k / QLk horse-race; Rivers-Vuong
  data/
    dominicks.py       # loader for qbz506/dreamprice-dominicks-cso
    panel.py           # generic panel schema + validation
    grid.py            # price-grid construction from empirical support
    markets.py         # market definition (store x week x category), competitor sets
  viz/
    phase.py, branch.py, response.py, currents.py
  cli.py               # `thermoqre solve|decompose|estimate|diagnose`
```

### 8.3 The central abstraction: `PayoffOracle`

This is what makes the library general-purpose across pricing domains and is the single most important design decision.

```python
from typing import Protocol
import jax.numpy as jnp


class PayoffOracle(Protocol):
    """Maps a joint price profile (and optional state) to per-firm profit.

    Implementations wrap ANY demand model. The library never assumes a
    functional form for demand; it only requires this map and, for gradient-
    based paths, that it be JAX-differentiable.
    """

    n_firms: int

    def profit(
        self,
        prices: jnp.ndarray,  # (n_firms,) or (batch, n_firms)
        state: jnp.ndarray | None = None,  # (state_dim,) market covariates
    ) -> jnp.ndarray:  # (n_firms,) or (batch, n_firms)
        ...

    def quantity(self, prices: jnp.ndarray, state: jnp.ndarray | None = None) -> jnp.ndarray: ...

    def elasticity(
        self, prices: jnp.ndarray, state: jnp.ndarray | None = None
    ) -> jnp.ndarray:  # (n_firms, n_firms) own+cross elasticity matrix
        ...
```

`PricingGame` then materialises the payoff tensor (or a matrix-free closure) by evaluating the oracle on the Cartesian product of per-firm price grids, `jax.vmap`-ed. For $N=2$, $m=100$ this is $10^4$ oracle calls — trivially batched. For $N=4$, $m=50$ it is $6.25\times10^6$ — still feasible batched on GPU, but the matrix-free path should be used for the response operators.

**DreamPrice adapter contract.** The adapter must expose the frozen causal demand decoder (`theta * log(price) + MLP(z_t, store_features)`) as `quantity`, combine with the cost basis to give `profit`, and expose the frozen elasticity as `elasticity`. Because DreamPrice is PyTorch and `thermoqre` is JAX, the adapter runs the torch model under `jax.pure_callback` (correct but blocks JIT/grad) **or**, preferred, exports the decoder weights to a small JAX re-implementation (the causal decoder is a 3-layer MLP plus a log-price term — trivially portable). **Port the decoder; do not callback.** Keep a callback path behind a flag for validating the port against the torch original.

### 8.4 Key function signatures (contract for the agent)

```python
# solve
def logit_qre(game, lam, *, method="mirror", init=None, tol=1e-10,
              max_iter=10_000) -> QREPoint: ...
def logit_branch(game, lam_max, *, n_points=200, first_step=0.03,
                 max_accel=1.1, detect_bifurcations=True) -> Branch: ...
def solve_at_lambdas(game, lams) -> list[QREPoint]: ...

# decompose
def hodge_decompose(game) -> tuple[Game, Game, Game]:  # (potential, harmonic, nonstrategic)
def alpha(game) -> float: ...
def make_family(base_potential, base_harmonic, alphas) -> list[Game]: ...

# response  -- the flagship objects
def chi_partial(qre_point) -> jnp.ndarray:            # block-diag lambda_i C_i
def chi_equilibrium(game, qre_point, *, matrix_free=False) -> jnp.ndarray:
    """(I - S B)^{-1} S  -- Proposition 1."""
def reciprocity_defect(game, qre_point) -> float:
    """||chi_eq - chi_eq^T||_F / ||chi_eq + chi_eq^T||_F -- Proposition 2."""
def strategic_spectrum(game, qre_point) -> SpectrumInfo:
    """eigenvalues of S B; rho; distance to criticality; bifurcation type."""
def critical_lambda(game, *, bracket=(0.0, 1e3)) -> float: ...

# dynamics
def glauber_generator(game, lam) -> sparse generator matrix
def stationary(generator) -> jnp.ndarray
def probability_currents(generator, pi) -> jnp.ndarray
def entropy_production_rate(generator, pi) -> float
def epr_from_trajectory(traj, *, method="kld", order=2) -> EPREstimate
def tur_bound(traj, current_fn) -> float

# estimate
def estimate_lambda_mle(game, counts, **kw) -> LambdaFit
def estimate_lambda_ccp(panel, oracle, **kw) -> LambdaFit
def estimate_lambda_dispersion(panel, oracle, **kw) -> LambdaFit
def estimate_lambda_hierarchical(panel, oracle, groups, **kw) -> LambdaFit
def compare_models(panel, oracle, models=("nash","qre","levelk","qlk")) -> ComparisonTable
```

`QREPoint` carries `sigma` (list of per-firm distributions), `lam`, `U` (expected payoffs), `residual`, `n_iter`, and lazily-computed `chi_part`, `chi_eq`, `R`, `spectrum`.

### 8.5 Numerics — non-negotiable requirements

- All softmax/logsumexp via `jax.nn.log_softmax` / `jax.scipy.special.logsumexp`. Never exponentiate raw payoffs.
- Payoffs internally rescaled to unit range with the scale factor folded into λ, and reported in both raw and normalised units — λ is not scale-free and users will misinterpret it otherwise. Expose `lambda_normalised = lam * payoff_range`.
- $C_i = \operatorname{diag}(\sigma_i) - \sigma_i\sigma_i^\top$ is rank-deficient by construction. All linear algebra on $(I-SB)$ must be done **on the tangent space** $T$ (mean-zero subspace), using an explicit orthonormal basis (Helmert or QR of $I - \frac{1}{m}\mathbf{1}\mathbf{1}^\top$). Getting this wrong silently produces a spurious zero eigenvalue and a false "criticality" reading. Unit-test the projection.
- Matrix-free path: `Lineax` GMRES on $v \mapsto v - S(Bv)$, where $Bv$ is computed by a `vmap`-ed contraction against the payoff tensor without materialising $B$.
- Implicit differentiation via the same operator — the VJP of the fixed point requires exactly $(I - SB)^{-\top}$, so the resolvent solve is shared between susceptibility and gradients. Implement once, reuse.
- Near criticality, warn the user: expose `distance_to_criticality = 1 - rho(SB)` and refuse to report $\chi^{\text{eq}}$ without a warning when it drops below $10^{-3}$.

### 8.6 Test plan

Exact-identity tests (property-based where possible):
1. $\nabla\psi = \lambda\sigma$ and $\nabla^2\psi = \lambda^2 C$ to $10^{-12}$.
2. Potential game ⟹ QRE marginals match Gibbs marginals of $Z^{-1}e^{\lambda\Phi}$ to $10^{-10}$.
3. Potential game ⟹ $\mathcal{R} = 0$, $J^* = 0$, EPR $= 0$, spectrum of $SB$ real.
4. Adding a nonstrategic component leaves $\sigma^*$ invariant.
5. `chi_equilibrium` matches finite-difference of `logit_qre` w.r.t. $h$ to $10^{-6}$.
6. `logit_branch` matches `pygambit.qre.logit_solve_branch` on 2×2 and 3×3 games to $10^{-8}$.
7. `estimate_lambda_mle` matches `pygambit.qre.logit_estimate` on simulated play counts.
8. $\lambda\to\infty$ limit of the principal branch converges to a Nash equilibrium (check via `pygambit`).
9. $\lambda\to 0$ gives the uniform centroid.
10. `epr_from_trajectory` recovers the exact EPR on simulated Glauber trajectories from a known non-potential game, within CI.

---

## 9. API service

**Framework:** FastAPI + Pydantic v2 + Uvicorn. **Deployment:** Render (Docker web service). **Async work:** Redis-backed queue (`arq` or RQ) + a Render background worker. **Persistence:** Render Postgres for jobs, results metadata, and user datasets; object storage (Render disk or S3-compatible) for uploaded panels and result artefacts.

Why a queue: branch continuation, entropy-production estimation, and hierarchical λ estimation are minutes-scale. Anything above ~5 s goes async.

### 9.1 Endpoints

```
POST /v1/games/from-matrix          -> game_id      (payoff tensors)
POST /v1/games/from-demand          -> game_id      (demand spec + price grids)
GET  /v1/games/{id}

POST /v1/decompose                  -> {alpha, norms, components_ref}
POST /v1/solve/qre                  -> QREPoint                (sync, small)
POST /v1/solve/branch               -> job_id                  (async)
POST /v1/response                   -> {chi_part, chi_eq, R, spectrum, rho,
                                        distance_to_criticality, bifurcation_type}
POST /v1/dynamics/stationary        -> {pi, currents, epr}     (async if large)
POST /v1/dynamics/epr-from-data     -> job_id                  (async)

POST /v1/datasets                   -> dataset_id  (upload CSV/Parquet panel)
GET  /v1/datasets/{id}/schema       -> inferred column mapping + validation report
POST /v1/estimate/demand            -> job_id      (loglog | logit | blp | dreamprice)
POST /v1/estimate/lambda            -> job_id      (mle | ccp | dispersion | hierarchical)
POST /v1/diagnose                   -> job_id      (full report: lambda, R, EPR bound,
                                                    phase location, model comparison)
POST /v1/optimize/price             -> {p_star, expected_profit, competitor_distribution,
                                        vs_nash, vs_myopic}

GET  /v1/jobs/{id}                  -> {status, progress, result | error}
GET  /v1/jobs/{id}/stream           -> SSE progress stream

GET  /v1/examples                   -> curated teaching games + preloaded scenarios
GET  /v1/health
```

### 9.2 Contract notes

- All numeric payloads are JSON arrays with explicit `shape` and `dtype`; large arrays returned as signed URLs to Parquet/NPZ, not inline.
- Every response embeds a `provenance` block: library version, oracle used, λ parameterisation, payoff normalisation constant, and a `warnings` list (e.g. `near_criticality`, `low_support`, `weak_instruments`).
- **`/v1/optimize/price` must return the competitor distribution alongside the point recommendation**, because the entire economic argument of §12 of `research1.rtf` is that the distribution over *rivals'* prices is the useful object, not a randomised own price. The UI must show both.
- Rate limiting and an API-key model from day one (`X-API-Key`), with a free tier keyed to small games.
- OpenAPI schema auto-published; ship a generated TypeScript client for the frontend and a thin `thermoqre-client` Python package.

### 9.3 Render specifics

- `render.yaml` blueprint declaring: `web` (Docker, FastAPI), `worker` (same image, different command), `redis`, `postgres`.
- Health check path `/v1/health`. Set `PORT` from env. Preload JAX on startup and warm the JIT cache with a tiny game so first user request isn't 20 s.
- Pin CPU-only JAX in the deployed image (`jax[cpu]`); GPU is a local/research concern, not a hosting one. Size the game limits accordingly and enforce them in validation.

---

## 10. Application

**Frontend:** Next.js (App Router) + TypeScript on Vercel. Tailwind + shadcn/ui. Charts: visx or Plotly (Plotly for 3-D phase surfaces and heatmaps; visx for everything else). KaTeX for maths. State: TanStack Query against the API.

Three modes, deliberately sequenced so a visitor can go from "what is QRE" to "here is λ for my own data" without leaving the app.

### 10.1 Learn — the teaching layer

Interactive, self-contained explainers that build the concept stack in order. Each is a page with live controls, not static prose.

1. **Softmax and λ.** Payoff bars → probability bars, with a λ slider from 0 to ∞. Shows the collapse to Nash. Directly renders the example from `research1.rtf` §4.
2. **Fixed point.** Two firms responding to each other; animate the iteration $\sigma_B \to EU_A \to \sigma_A \to EU_B$ until convergence. Show the map and the fixed point.
3. **QRE vs mixed Nash.** Side-by-side: mixed Nash requires indifference; QRE does not. Show $EU(A)=10$, $EU(B)=8$, $P(B)>0$.
4. **MaxEnt.** Slider on the entropy-regularisation weight $1/\lambda$; show that maximising $\mathbb{E}[U] + \lambda^{-1}H$ *produces* the logit. Show the free energy $F = U - TS$ decomposition.
5. **Gibbs and potential games.** Build a small congestion/coordination game, display $\Phi$, show QRE marginals coinciding with the Gibbs measure. Then break potentiality and watch them separate.
6. **Detailed balance and currents.** Animate the Glauber chain on the profile lattice; show $J^*=0$ (potential) vs. circulating $J^*$ (harmonic). This is the visual heart of the whole programme.
7. **Reciprocity.** Poke firm 1 with a cost shock, measure firm 2's response; poke firm 2, measure firm 1. Show the two numbers coinciding in a potential game and diverging as α rises. Then show $\mathcal{R}$ tracking α.
8. **Elasticity vs. λ.** The `research1.rtf` hierarchy: demand → payoff → game → QRE. Show that changing elasticity changes the payoff surface and hence the QRE, while λ only changes response sharpness. This directly forestalls the "QRE implicitly models demand" confusion.
9. **The one-price objection.** Reproduce the PI's own critique honestly: show that if you just want tomorrow's price you should argmax expected profit — and then show that the expected profit is computed against the *competitor distribution*, which is what QRE supplies. This section is the app's intellectual credibility.

### 10.2 Lab — the synthetic experiment

The α-sweep from Stage 2, made interactive.

- Game builder: pick a template (Bertrand duopoly, capacity-constrained, coordination, congestion) or supply a payoff tensor.
- Two master sliders: **λ** (inverse temperature) and **α** (harmonic fraction). Everything below updates live.
- Panels: price distribution per firm; QRE branch with turning points marked; spectrum of $SB$ in the complex plane with the unit circle drawn (bifurcation type read directly off it); $\mathcal{R}$ gauge; probability-current field on the profile lattice; entropy production readout; a running phase-diagram heatmap in $(\lambda,\alpha)$ with the user's current position pinned.
- "Anneal" button: run a λ-quench and show the Jarzynski/Crooks work distribution — this is the demo that makes the thermodynamics visceral.
- Export: every panel exports to PNG/SVG and the underlying arrays to CSV; a "reproduce this in Python" button emits a runnable `thermoqre` snippet. **This last feature is the main funnel from app to library.**

### 10.3 Analyze — bring your own data

The commercial/practical layer, and the reason this is a product rather than a demo.

Flow:
1. **Upload** a panel (CSV/Parquet) or connect the bundled Dominick's example. Minimum schema: entity id, time, price, quantity. Optional: cost, promotion flag, competitor id/price, market id, covariates.
2. **Schema mapping** with inference and a validation report (missingness, price-grid density, panel balance, sufficient cross-sectional variation for the chosen estimator). Refuse loudly and specifically when the data cannot identify what the user is asking for.
3. **Demand stage.** Choose oracle: constant-elasticity (fast, always available), logit/nested logit, BLP (if instruments supplied), or upload/point to a custom model. Report elasticities with CIs and a first-stage strength diagnostic. **Gate everything downstream on this passing.**
4. **Strategic stage.** Estimate λ (all four estimators, shown together with agreement/disagreement flagged), $\mathcal{R}$, TUR bound on dissipation, and the phase-diagram location of this market.
5. **Decision stage.** For a chosen firm: the competitor price distribution, the expected-profit curve over the user's price grid, $p^*$, and a comparison against Nash-Bertrand, myopic cost-plus, and match-competitor benchmarks — with out-of-sample backtest scores if enough history exists.
6. **Report.** A downloadable HTML/PDF containing the full analysis, every assumption made, every warning raised, and a plain-language interpretation section. This is what the user actually takes to their team.

**Non-negotiable product ethics.** The report must state, prominently, that λ absorbs unmodelled heterogeneity; that a low λ is not proof of irrationality; that the recommendation is conditional on the demand model; and that this is decision support, not an autonomous pricing system. Carry over the DreamPrice model card's "NOT intended for" list.

### 10.4 Vercel specifics

- Static/ISR for Learn (content rarely changes; prerender). Client-side data fetching for Lab and Analyze.
- Heavy compute never runs in Vercel functions — it all goes to the Render API. Vercel functions only proxy auth and short-lived requests.
- Small in-browser WASM path (optional, later): a tiny pure-TS logit solver for the Learn sliders so the teaching layer is instant and works offline. Do not attempt this for Lab or Analyze.
- Upload flow: signed direct-to-storage upload; Vercel never holds the file.

---

## 11. Build order for the agent

Strictly sequential; each step ends with green tests before the next begins.

**Phase 1 — library core (no data, no service)**
1. Scaffold package, `pyproject.toml`, CI, float64 config, `types.py`.
2. `games/tensor.py` + `games/library.py` with the canonical test games, including at least one *verified* exact potential game and one pure harmonic game (RPS).
3. `solve/fixedpoint.py` (damped + Anderson) and `solve/mirror.py`. Tests 1, 9.
4. `solve/validate.py` against `pygambit`; `solve/homotopy.py`. Tests 6, 8.
5. `response/susceptibility.py` with the tangent-space projection done correctly. Tests 5.
6. `response/reciprocity.py`, `response/spectral.py`. Test 3 (potential ⟹ $\mathcal{R}=0$, real spectrum).
7. `decompose/hodge.py` + `decompose/generate.py`. Test 4 (nonstrategic invariance).
8. `dynamics/*` — generator, stationary, currents, exact EPR. Test 2, 3, 10.
9. `solve/implicit.py` reusing the resolvent.

**Phase 2 — pricing and estimation**
10. `demand/oracle.py` protocol; `demand/loglog.py`; `games/factored.py` (PricingGame).
11. `demand/dreamprice.py` — port the causal decoder to JAX, validate against torch.
12. `data/*` — Dominick's loader, grid construction, market definition.
13. `estimate/mle.py`, then `ccp.py`, `dispersion.py`, `hierarchical.py`, `compare.py`. Test 7.
14. `cli.py`, docs, examples notebooks.

**Phase 3 — API**
15. FastAPI app, Pydantic schemas mirroring the library types, sync endpoints.
16. Job queue + worker + Postgres; async endpoints; SSE progress.
17. `render.yaml`, Dockerfile, JIT warm-up, limits and validation, API keys, rate limits.
18. OpenAPI → generated TS client; `thermoqre-client` Python package.

**Phase 4 — App**
19. Next.js scaffold, design system, KaTeX, chart primitives.
20. Learn (all nine explainers) — ship this first; it is the lowest-risk, highest-credibility surface.
21. Lab — the α/λ sliders and all live panels.
22. Analyze — upload, schema mapping, staged pipeline, report generation.
23. Deploy: Render (API) + Vercel (app), custom domain, analytics, error tracking.

**Phase 5 — science runs**
24. Stage 1 and Stage 2 experiments executed *through the library*, results checked into a `experiments/` directory with fixed seeds and a `make reproduce` target.
25. Stage 3 empirics.

---

## 12. Repository layout

```
thermoqre/                      # monorepo
  packages/
    thermoqre/                  # the library (published to PyPI)
    thermoqre-client/           # thin Python API client
  services/
    api/                        # FastAPI + worker; Dockerfile; render.yaml
  apps/
    web/                        # Next.js; vercel.json
  experiments/
    stage1_potential/           # verification suite
    stage2_alpha_sweep/         # the phase diagram
    stage3_dominicks/           # empirics
    Makefile                    # `make reproduce`
  papers/
    p1_reciprocity/  p2_nonequilibrium/  p3_empirical/
  docs/                         # mkdocs-material; theory notes = app Learn content source
  CLAUDE.md                     # agent working notes, conventions, gotchas
```

Keep `docs/theory/*.md` as the single source of truth for both the documentation site and the app's Learn content — write the explainers once, render twice.

---

## 13. Risk register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Proposition 2 already exists in the literature | Medium | Stage 0 verification before any writing; software value is unaffected |
| 2 | Tangent-space projection bug produces phantom criticality | High | Dedicated unit tests; refuse to report χ near singularity; cross-check vs finite differences |
| 3 | Hodge decomposition intractable at realistic grid sizes | High | Exact path for small games; pairwise-marginal approximation for large; document the approximation error |
| 4 | DreamPrice torch→JAX port introduces silent numerical drift | Medium | Keep the callback path; assert agreement to 1e-6 on a fixed test batch in CI |
| 5 | λ not separately identified from collusion / misspecification | High | Present as bounded conclusions with sensitivity bands; foreground the λ-free reciprocity test |
| 6 | Dominick's rival-firm interpretation is weak | High | State it everywhere; treat as an illustrative anchor, not the evidential foundation; design the app so users bring genuinely competitive data |
| 7 | Entropy-production estimators unreliable under partial observation | Medium | Report the TUR *lower bound* as headline; validate all estimators on synthetic ground truth first |
| 8 | "Physics metaphor" reviewer attack | Medium | The tiering in §3 is the defence: A-tier exact, Props 1–2 proved, everything else labelled. Never let a decorative analogy into a claim |
| 9 | Scope explosion (library + API + app + 3 papers) | High | Build order in §11 is strictly sequential; Learn ships before Lab, Lab before Analyze; each phase is independently useful |
| 10 | Hosting cost of JAX workloads on Render | Low | CPU-only image, enforced game-size limits, async queue, aggressive result caching keyed on game hash + λ |

---

## 14. Immediate next actions

1. **Stage 0 literature check on Proposition 2** — highest priority, cheapest, and determines paper framing.
2. Scaffold `packages/thermoqre` and implement Phase 1 steps 1–6. The first genuinely new artefact in the world is `reciprocity_defect()` returning zero on a potential game and rising with α.
3. Port the DreamPrice causal decoder to JAX (Phase 2 step 11) in parallel — it is independent of everything else and unblocks all empirics.
4. Write `docs/theory/01_softmax_and_lambda.md` through `09_the_one_price_objection.md` as the shared source for docs and the app's Learn mode.
