# ThermoQRE v3 — Research and Build Specification

**Status:** Master context document. Supersedes `research1.rtf`, the v1 research report, and v2.
**Audience:** (a) the principal investigator, (b) a Claude Code agent building the software.
**Author context:** PI is a Python expert, game theorist, and non-equilibrium thermodynamicist. Prior work: DreamPrice (`github.com/SharathSPhD/dreamprice`, `hf.co/qbz506/dreamprice-cso`).
**Licence intent:** library Apache-2.0; app/API source-available; Dominick's-derived artefacts remain CC-BY-NC-4.0.

---

## 0. How to work on this

### 0.1 Working philosophy — read this before anything else

This is **not** a pre-registered hypothesis-testing programme. Earlier drafts of this document were written in that idiom — kill criteria, falsification thresholds, stage gates — and that idiom is wrong for this project. Discard it.

The right model is **exploratory instrumentation building**. We are constructing a set of measuring devices for strategic pricing systems — a susceptibility meter, a reciprocity meter, an entropy-production meter, a phase locator — and then pointing them at things to see what they read. The interesting outcomes will mostly be ones nobody wrote down in advance.

Concretely, this means:

- **Build the instrument first, decide what it means second.** If `reciprocity_defect()` returns a number, that number is interesting regardless of whether it matched a prediction. Go look at *why* it's that number.
- **Anomalies are the product, not the failure mode.** If the susceptibility blows up somewhere unexpected, that is a lead, not a bug report against a hypothesis. Chase it.
- **No stage gates.** Do not block Stage 3 on Stage 2 "passing." Work on whatever is currently unblocked and interesting. The empirical work and the theory work inform each other continuously; serialising them wastes the feedback.
- **Ship working code over correct proofs.** A function that computes the right number on a test case you trust is worth more than a theorem you haven't implemented. Prove things when the proof would change what you build.
- **When theory and numerics disagree, assume the theory is incomplete.** Nine times out of ten the numerics have found a case the derivation didn't cover. Go find the case.
- **The app is a research instrument, not a demo.** Every panel in the Lab is an experiment you can run in ten seconds. Build it early and use it constantly. Most of the discoveries will come from sliding sliders and noticing something.
- **Negative or null results are fine and unremarkable.** They don't need to be dressed up as falsification. If λ estimates disagree on Dominick's, that's a finding about Dominick's and about the estimators; write it down and move on.
- **Don't gold-plate the honesty caveats.** State limitations once, clearly, where they belong. Repeating them ritually is not rigour, it's noise.

The scientific claims in §3 are labelled by confidence — **known**, **derived-and-checked**, **conjectured**, **speculative** — so you know what you're standing on. Update those labels freely as evidence arrives. That is the whole epistemic apparatus this project needs.

### 0.2 Structure

§1–§6 are science and context. §7–§14 are the build. When an engineering choice is ambiguous, resolve it in favour of whatever makes the §3 objects directly computable and directly visible in the app.

Two standing rules for the agent:
1. **Do not write half-files.** Write a module fully, or edit with targeted replacements. Never assume unseen code exists.
2. **Every named quantity in §3 becomes a named, tested function.** No exceptions. That discipline is what makes this more than Gambit.

---

## 1. Sanity-check results (done — read before claiming anything)

Two checks were run against the literature. Both changed the picture.

### 1.1 On the reciprocity ⟺ potentiality claim: partly known, and the precise statement needs fixing

**What is already established (do not claim):**

- The condition that the cross-payoff operator be symmetric — $\partial U_i(a)/\partial\sigma_j(b) = \partial U_j(b)/\partial\sigma_i(a)$ — is **Sandholm's externality symmetry**, the standard integrability characterisation of potential games in population-game form. Sandholm, "Potential Games with Continuous Player Sets," *JET* 97(1):81–108 (2001); *Population Games and Evolutionary Dynamics*, MIT Press (2010), Ch. 3.
- The Jacobian-splitting version — decompose the game's individual-gradient Jacobian into symmetric + antisymmetric parts, potential iff antisymmetric part vanishes, "Hamiltonian" iff symmetric part vanishes — is Balduzzi, Racanière, Martens, Foerster, Tuyls & Graepel, "The Mechanics of *n*-Player Differentiable Games," ICML 2018 (the SGA / Helmholtz-decomposition paper).
- Candogan, Menache, Ozdaglar & Parrilo (2011) give the intrinsic finite-game version via the flow decomposition.

So **"B symmetric ⟺ potential" is textbook.** The v2 draft claimed it as new. It is not. Correct that in any writing.

**Important precision fix.** Full externality symmetry ($v = d\phi$ on the *full* payoff field) is **sufficient but not necessary** for a finite game to be a potential game — a finite normal-form game is potential iff its *effective* payoff field is exact, which is weaker. See Ramaswamy/Mertikopoulos et al., "A Geometric Decomposition of Finite Games: Convergence vs. Recurrence under Exponential Weights," arXiv:2405.07224, Lemma C.2 and Example C.2. **Therefore the correct statement of the symmetry condition is that it characterises *full* potential games, and on the tangent space / normalised game it characterises potential games.** Get this right or it is the first thing a referee kills. The library must decompose on the *normalised* (strategically equivalent, effective) game, not the raw payoff tensor.

**What appears to survive as genuinely new:**

- **(N1) The resolvent transfer.** $\chi^{\text{eq}} = (I - SB)^{-1}S$ is symmetric **iff** $S(B - B^\top)S = 0$. That is, the strategic feedback loop neither creates nor destroys reciprocity — the *observable equilibrium* response matrix inherits the symmetry of the *unobservable* payoff operator exactly. This is not obvious: one would generically expect a resolvent to break symmetry. I could not find this stated.
- **(N2) $\mathcal{R}$ as an observable.** Because of N1, the reciprocity defect of the measured cross-response matrix is a proxy for harmonic content, estimable from cross-firm cost-shock pass-through asymmetry without knowing payoffs. That measurement framing appears new.
- **(N3) No Hopf in potential games.** $B = B^\top \Rightarrow SB \sim S^{1/2}BS^{1/2}$ symmetric $\Rightarrow$ real spectrum $\Rightarrow$ no complex pair crossing $\Rightarrow$ no Hopf bifurcation. So logit dynamics in a (full) potential game cannot produce sustained price cycles. Plausibly folklore, but I did not find it stated for logit/QRE specifically.

**Also worth knowing:** the behaviour of learning dynamics as games move from potential toward harmonic "hit an important obstacle, and has remained an open question since the original work of Candogan et al." — per arXiv:2405.07224, which is the current state of the art on exactly the α-sweep territory. **Read that paper properly before designing the sweep.** It is either a collaborator-shaped result or a duplication risk, and it is recent enough that the PI's non-equilibrium-thermodynamics angle (currents, entropy production, TUR) is plausibly the differentiator.

**Net effect on the programme:** the theory contribution is narrower than v2 claimed but still real, and it is now precisely located. The *instrument* — a working, fast, differentiable reciprocity/susceptibility/dissipation meter that nobody currently has — is unaffected and is arguably the bigger contribution anyway.

### 1.2 On Hodge tractability: much better than feared

v2 flagged the decomposition as a scaling risk (Risk #3, "High"). That was pessimistic, because it ignored the structure of the game graph.

The game graph for the flow decomposition is the **Cartesian product of complete graphs**, $K_{m_1} \,\square\, K_{m_2} \,\square\cdots\square\, K_{m_N}$. Cartesian products of graphs have Laplacians that are Kronecker sums:
$$L = \bigoplus_{i} L_{K_{m_i}} = \sum_i I \otimes\cdots\otimes L_{K_{m_i}} \otimes\cdots\otimes I,$$
and $L_{K_m} = mI - \mathbf{1}\mathbf{1}^\top$ has a **closed-form eigendecomposition**: eigenvalue $0$ on $\mathbf{1}$, eigenvalue $m$ with multiplicity $m-1$ on the mean-zero subspace. Consequently:

- The eigenbasis of $L$ is a tensor product of trivially-known bases; eigenvalues are sums $\sum_i m_i \cdot \mathbb{1}[\text{component } i \text{ non-constant}]$.
- The Hodge projections (potential / harmonic / nonstrategic) reduce to **projections onto tensor-product eigenspaces indexed by which coordinates are "non-constant."** This is a separable, FFT-like transform, not a generic least-squares solve.
- Cost is $O(\text{payoff tensor size} \times \sum_i m_i)$ with no iterative solver, no matrix materialisation, and it `vmap`s cleanly.

Sizes: $N=2, m=100 \Rightarrow 2\times10^4$ entries (instant). $N=3, m=50 \Rightarrow 3.75\times10^5$ (instant). $N=4, m=50 \Rightarrow 2.5\times10^7$ (seconds on GPU, fine). $N=5, m=30 \Rightarrow 1.2\times10^8$ (needs care but feasible).

**Downgrade Risk #3 from High to Low, and implement the separable transform rather than a generic projection.** This is the single most valuable engineering finding from the sanity checks.

**One caveat carried forward:** do not build α by adding a "rotational component" to the Jacobian. The skew-symmetric part of a Jacobian is coordinate-dependent and generally not integrable (arXiv:2405.07224). α must come from the intrinsic Candogan flow decomposition. The spec already does this; keep it that way.

---

## 2. What this project is

**The through-line.**
> Build instruments that measure how far a strategic pricing system is from thermodynamic equilibrium, and see what they read — on synthetic games where we control the answer, and on real scanner data where we don't.

Three regimes, used as a *map* rather than a set of hypotheses to adjudicate:

- **Potential games** — the calibration standard. Everything is analytically known: QRE = Gibbs, zero current, zero dissipation, symmetric response. Use these to check the instruments read zero when they should.
- **Near-potential games** — the interesting middle. Sweep the harmonic fraction α and watch every meter move. Most of the discoveries live here.
- **Generic pricing games** — the target. Currents, dissipation, cycling, multiplicity.

**Three things to look at** (framed as directions of attention, not hypotheses to test):

1. *Calibration.* Do the meters read zero on exact potential games? If not, why not — implementation, or something the theory missed?
2. *The α-sweep.* How do $\mathcal{R}$, $\|J^*\|$, $\sigma_{\text{EP}}$, spectral structure, and cycle amplitude co-vary as potentiality breaks? Is the relationship smooth, threshold-like, or something else? Is there a regime where they decouple?
3. *Contact with data.* What do the meters read on real scanner panels, and how much of the reading is robust to the demand model underneath?

**Explicitly not the point.** Beating a tuned demand-based optimiser at setting tomorrow's price. The app measures it because users will ask, but nothing here depends on winning that comparison.

---

## 3. Theory, labelled by confidence

Notation. Players $i \in \{1,\dots,N\}$ (firms). Finite action sets $A_i$ (price grids), $|A_i| = m_i$. Mixed strategies $\sigma_i \in \Delta(A_i)$, full support. Expected payoff to pure action $a$:
$$U_i(a;\sigma_{-i}) = \sum_{a_{-i}} \Big(\prod_{j\ne i}\sigma_j(a_j)\Big)\, u_i(a, a_{-i}).$$
Logit QRE with precision $\lambda_i$:
$$\sigma_i(a) = \frac{\exp\{\lambda_i U_i(a;\sigma_{-i})\}}{\sum_{b\in A_i}\exp\{\lambda_i U_i(b;\sigma_{-i})\}}.$$

### 3.1 KNOWN — implement as verified primitives, cite, never claim

**(K1) Entropy-regularised best response.** $\sigma_i = \arg\max_{\sigma\in\Delta(A_i)}\{\mathbb{E}_\sigma[U_i] + \lambda_i^{-1}H(\sigma)\}$. Gibbs variational principle; Fudenberg–Iijima–Strzalecki (*Econometrica* 2015) for the perturbed-utility view.

**(K2) Log-partition is the CGF.** With $\psi_i = \log\sum_a e^{\lambda_i U_i(a)}$:
$$\nabla_U\psi_i = \lambda_i\sigma_i,\qquad \nabla^2_U\psi_i = \lambda_i^2 C_i,\qquad C_i := \operatorname{diag}(\sigma_i)-\sigma_i\sigma_i^\top.$$
Williams–Daly–Zachary + exponential family.

**(K3) Gibbs measure in potential games.** Exact potential $\Phi$ ⟹ Glauber/logit dynamics reversible with $\pi(a) = Z^{-1}e^{\lambda\Phi(a)}$; detailed balance; $J^*=0$; $\sigma_{\text{EP}}=0$. Blume (1993); Monderer–Shapley (1996).

**(K4) Externality symmetry ⟺ (full) potential.** $B = B^\top$ on the appropriate space characterises full potential games. Sandholm (2001, 2010 Ch. 3); Balduzzi et al. (ICML 2018). **Note the full-vs-effective distinction in §1.1.**

**(K5) Hodge/flow decomposition.** $\mathcal{G} = \mathcal{G}_{\text{pot}} \oplus \mathcal{G}_{\text{harm}} \oplus \mathcal{G}_{\text{nonstrat}}$, orthogonal direct sum. Candogan, Menache, Ozdaglar & Parrilo, *MOR* 36(3):474–503 (2011). Nonstrategic components do not affect QRE at all — a free invariant to test against.

**(K6) Rational-inattention foundation for λ.** Matějka–McKay (*AER* 2015); Fosgerau–Melo–de Palma–Shum (*IER* 2020). λ is the inverse shadow price of Shannon information.

**(K7) Partial susceptibility.** With field $h_i \in \mathbb{R}^{m_i}$ added to own payoffs,
$$\chi^{\text{part}}_i := \frac{\partial\sigma_i}{\partial h_i}\Big|_{\sigma_{-i}} = \lambda_i C_i, \qquad \frac{\partial\mathbb{E}[p]}{\partial\eta}\Big|_{\sigma_{-i}} = \lambda_i\operatorname{Var}_{\sigma_i}(p).$$
Exact static FDT — but opponents-frozen, so not what data shows.

### 3.2 The canonical α

Define the **harmonic fraction** on the *normalised* (effective) game:
$$\alpha(u) := \frac{\|u^{H}\|}{\|u^{P}\| + \|u^{H}\|} \in [0,1].$$
Intrinsic, basis-independent, and — per §1.2 — computable by a separable tensor transform in near-linear time. Synthetic families are generated by projecting and re-mixing at target α, never by perturbing a Jacobian.

### 3.3 DERIVED-AND-CHECKED — the resolvent

Let $B_{ij}(a,b) := \partial U_i(a;\sigma_{-i})/\partial\sigma_j(b)$ for $j\neq i$, $B_{ii}:=0$; let $S := \operatorname{blockdiag}(\lambda_i C_i) = \chi^{\text{part}}$, symmetric PSD, positive definite on the tangent space $T=\bigoplus_i\{v:\mathbf{1}^\top v=0\}$ under full support.

Total differentiation of the fixed point gives $d\sigma = S(dh + B\,d\sigma)$, hence:

> **Result 1 (strategic resolvent).** Where $(I-SB)$ is invertible on $T$,
> $$\chi^{\text{eq}} := \frac{d\sigma^*}{dh} = (I - SB)^{-1}S = (I - \chi^{\text{part}}B)^{-1}\chi^{\text{part}}.$$

Straightforward comparative statics; likely folklore in some form. Cite defensively, don't claim.

> **Result 2 (N1 — reciprocity transfer).** $\chi^{\text{eq}}$ is symmetric $\iff S(B-B^\top)S = 0$, i.e. (full support) $\iff B = B^\top$ on $T$ $\iff$ the normalised game has zero harmonic component.
>
> *Proof.* Symmetry requires $(I-SB)^{-1}S = S(I-B^\top S)^{-1}$. Multiply through: $S(I-B^\top S) = (I-SB)S$, i.e. $S - SB^\top S = S - SBS$, i.e. $S(B-B^\top)S = 0$. Positive definiteness on $T$ removes the sandwich. □

The *content* of Result 2 is not the characterisation of potentiality (that's K4) but the **transfer**: strategic feedback preserves reciprocity exactly. That is what makes the defect measurable.

Define the **reciprocity defect**
$$\mathcal{R} := \frac{\|\chi^{\text{eq}} - (\chi^{\text{eq}})^\top\|_F}{\|\chi^{\text{eq}} + (\chi^{\text{eq}})^\top\|_F}.$$

Why it's worth building: it is (i) a physics-standard equilibrium signature — Onsager reciprocity, which fails in NESS and under broken time-reversal; (ii) an operational test for potentiality requiring no knowledge of $\Phi$; (iii) estimable from **cross-firm cost-shock pass-through asymmetry** — how much firm $i$ moves when firm $j$'s cost moves, versus the reverse; and (iv) **λ-free** as a symmetry statement, which is a useful property given Haile–Hortaçsu–Kosenok (§4.1).

### 3.4 The same operator gives the phase structure

$(I-SB)$ is simultaneously the susceptibility resolvent, the contraction certificate ($\rho(SB)<1$ ⟹ unique QRE, and since $S \propto \lambda$ this gives a small-λ uniqueness bound), and the bifurcation detector on the QRE correspondence (turning points are where an eigenvalue of $SB$ crosses 1).

| Spectral event | Reading | Signature |
|---|---|---|
| $\rho(SB)<1$ | unique diffuse QRE | single branch, bounded $\chi^{\text{eq}}$ |
| real eigenvalue crosses 1 | fold/pitchfork — multiplicity, coordination (Brock–Durlauf regime) | branch turning point, bimodal prices |
| complex pair crosses | Hopf — sustained price cycles | limit cycle, $J^*\neq0$, dissipation jump |

> **Result 3 (N3 — conjectured, easy to check).** $B=B^\top \Rightarrow SB \sim S^{1/2}BS^{1/2}$ symmetric $\Rightarrow$ real spectrum $\Rightarrow$ no Hopf. So price cycles cannot arise under logit dynamics in a full potential game.

Chain worth watching: cycling → non-potential → broken reciprocity → positive dissipation. Whether it holds tightly, loosely, or breaks somewhere is exactly the kind of thing the α-sweep is for.

### 3.5 Non-equilibrium layer

Glauber/logit Markov jump process on the joint profile space; at stationarity:
$$J^*(a,a') = \pi^*(a)w(a\to a') - \pi^*(a')w(a'\to a),$$
$$\sigma_{\text{EP}} = \tfrac12\sum_{a,a'}\big[\pi^*(a)w(a\to a') - \pi^*(a')w(a'\to a)\big]\log\frac{\pi^*(a)w(a\to a')}{\pi^*(a')w(a'\to a)} \geq 0.$$
Also implement: Hatano–Sasa housekeeping/excess split; TUR as a lower bound on dissipation needed to sustain a cycle of given regularity; Crooks/Jarzynski for λ-quench protocols.

For **real data**, exact rates are unavailable. Three estimators of irreversibility from observed price-profile trajectories:
1. **KLD / $k$-th order Markov** — $\hat\sigma^{(k)}=\frac{1}{k\tau}\sum P(Y_{0:k})\log\frac{P(Y_{0:k})}{P(Y_{k:0})}$. Assumption-light, data-hungry.
2. **TUR lower bound** — from first two cumulants of an empirical current. Certified bound rather than point estimate; **use this as the headline empirical number** because it degrades gracefully under partial observation.
3. **Neural (NEEP-style)** — variational, best in high dimension, watch overfitting.
Cross-check all three on synthetic trajectories with known $\sigma_{\text{EP}}$ before pointing them at anything real. References: Otsubo–Manikandan–Sagawa–Krishnamurthy, *Comms Physics* (2022), arXiv:2010.03852; Xu & Wang, arXiv:1107.6043 (which identifies an Edgeworth price cycle in experimental market data via non-zero EPR — the closest existing precedent, and the paper to read first).

### 3.6 λ as a structured object

$\lambda_{i,t} = \exp(x_{i,t}^\top\beta)$ with $x$ containing price dispersion, competitor volatility, time since last price change, promotion state, information-quality proxies. Hierarchical partial pooling across stores/categories. Decomposing λ into information cost / menu cost / forecast error / organisational heterogeneity is not separately identified without exclusion restrictions — say so once, propose candidates (e.g. store staffing turnover as an organisational shifter), and move on.

---

## 4. Two things people will push on

### 4.1 Haile–Hortaçsu–Kosenok (*AER* 98(1):180–200, 2008)

Unrestricted QRE rationalises anything. Three practical responses, all used:
1. Restrict to logit + regular-QRE axioms (Goeree–Holt–Palfrey). Necessary, weak alone.
2. **Externally identify payoffs.** DreamPrice's IV-identified demand pins down $u_i$, making QRE a low-dimensional restriction on the joint price distribution. This is the workhorse.
3. **Use the λ-free restriction.** Result 2 constrains the *symmetry* of the cross-response matrix independently of λ's value — nothing to absorb misfit. This is the most interesting angle and worth leading with.

### 4.2 λ versus collusion

Low λ and tacit collusion (Calvano–Calzolari–Denicolò–Pastorello, *AER* 110(10):3267–3297, 2020) look similar in level and dispersion. Things that might separate them, worth trying:
- Joint (mean, dispersion) signature — logit noise is roughly mean-preserving at leading order; collusion shifts the mean.
- $\mathcal{R}$ — coordinated conduct is largely symmetric; harmonic circulation is not.
- Dissipation — punishment phases are transient excursions; cycling is persistent circulation.
- Rivers–Vuong selection across the conduct menu, following Duarte–Magnolfi–Sølvsten–Sullivan, *Quantitative Economics* 15(3):571–606 (2024), with their weak-instrument diagnostic.
Separation will be partial. That's fine; report what separates and what doesn't.

---

## 5. Work streams (parallel, not staged)

Four streams. Run whichever is unblocked. Cross-pollinate constantly.

**Stream A — Instruments.** The library core: solvers, decomposition, susceptibility, reciprocity, spectrum, currents, dissipation. Everything else depends on this, so it starts first, but it never "finishes" — new meters get added as questions arise.

**Stream B — The synthetic lab.** Calibration on potential games; the α-sweep; the phase map in $(\lambda,\alpha,N,\text{cross-elasticity})$; quench protocols. Mostly done by *using the app*, which is why the Lab UI is built early rather than last.

**Stream C — Data contact.** DreamPrice adapter, Dominick's pipeline, λ estimators, $\mathcal{R}$ from pass-through asymmetry, TUR bounds. Independent of B; start as soon as the oracle protocol exists.

**Stream D — Product.** API and app. Learn mode first (lowest risk, immediately useful for the PI's own thinking and for explaining the work), then Lab (which is Stream B's instrument), then Analyze.

**Rough ordering of first moves, not a gate sequence:**
1. Read arXiv:2405.07224 properly. It is the nearest live work to Stream B.
2. Library scaffold + `reciprocity_defect()` working end-to-end on a potential game and a harmonic game. First new artefact in the world.
3. Separable Hodge transform (§1.2) — unblocks all of Stream B and is a self-contained, satisfying piece of code.
4. DreamPrice decoder ported to JAX — unblocks all of Stream C, independent of everything else.
5. Learn-mode explainers, which double as `docs/theory/`.

**Things worth noticing along the way** (not pass/fail criteria — just what to keep an eye on):
- Do the meters read zero on verified potential games? If not, is it the tangent-space projection, the normalisation, or something real?
- Is $\mathcal{R}(\alpha)$ smooth, or does it have structure — thresholds, plateaus, non-monotonicity?
- Does Result 3 hold, and if a cycle shows up at α=0, what is it?
- Do the three λ estimators agree anywhere, and where they diverge, does the divergence track anything interpretable?
- Does $\mathcal{R}$ measured from Dominick's pass-through asymmetry sit anywhere near what the synthetic games at comparable α produce?

---

## 6. Writing (a by-product)

- **Instruments paper / software note.** `thermoqre` as a measurement toolkit. Probably the highest-value output and the easiest to write. JOSS or *Journal of Statistical Software*.
- **Reciprocity paper.** Results 1–3, the spectral typology, the α-sweep. Honest positioning against Sandholm/Balduzzi/Candogan per §1.1. *GEB* / *JET* / *PRE*.
- **Non-equilibrium pricing paper.** Currents, dissipation, cycling, phase map. *PRE* / *JEDC*.
- **Empirical paper.** Dominick's + DreamPrice. *RAND* / *Quantitative Economics*.

Write them when there's something to say. Don't plan around them.

---

# BUILD SPECIFICATION

---

## 7. Architecture

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

### 7.1 Delta versus Gambit

Gambit 16.7.0 (`pygambit`) ships `logit_solve`, `logit_solve_branch`, `logit_solve_lambda`, and `logit_estimate` (MLE along the correspondence, `use_empirical` fast path). It is excellent. Use it as the ground-truth oracle; do not compete on correctness, inherit it.

| | pygambit | thermoqre |
|---|---|---|
| QRE correspondence, small finite games | ✅ authoritative | ✅ validated *against* Gambit |
| MLE of λ from play frequencies | ✅ | ✅ + CCP, dispersion, hierarchical, covariate-λ |
| Payoffs supplied as | a matrix you build | **a demand model + price grid** |
| Large price grids ($m\sim10^2$) | impractical | ✅ JAX, matrix-free |
| Differentiable through equilibrium | ✗ | ✅ implicit diff |
| Hodge decomposition, α | ✗ | ✅ separable transform |
| $\chi^{\text{eq}}$, $\mathcal{R}$ | ✗ | ✅ core |
| Currents, entropy production, TUR | ✗ | ✅ core |
| Bifurcation typing | partial | ✅ typed spectral classification |
| Panel pipeline, market definition | ✗ | ✅ |
| Hosted API + interactive app | ✗ | ✅ |

---

## 8. Library: `thermoqre`

### 8.1 Stack

Python ≥3.11. **JAX** with `jax_enable_x64` on by default — the susceptibility and dissipation computations are ill-conditioned near criticality and float32 is not acceptable. **Equinox** (module structure), **Optimistix** (root-find/fixed-point with implicit diff — note JAXopt is in maintenance mode and merging into Optax; do not build on JAXopt), **Lineax** (linear solves incl. matrix-free GMRES), **Optax** (first-order optimisers). **NumPyro** (hierarchical λ). **polars** (panels). Optional extras: `pygambit`, `pyblp`, `datasets`/`huggingface_hub`.

Packaging: `pyproject.toml` + hatchling. Extras `[gambit] [blp] [dreamprice] [bayes] [viz] [all]`. Tests: pytest + hypothesis. CI: GitHub Actions over 3.11/3.12/3.13.

### 8.2 Module map

```
thermoqre/
  types.py             # Game, QREPoint, Branch, Diagnostics (equinox.Module)
  games/
    base.py            # Game protocol
    tensor.py          # DenseTensorGame
    factored.py        # PricingGame from a PayoffOracle over a price grid
    normalise.py       # effective/normalised game (see §1.1 precision fix)
    library.py         # congestion (exact potential), RPS (harmonic),
                       # matching pennies, coordination, Bertrand-logit duopoly
  decompose/
    kron.py            # separable Kronecker eigenbasis for K_m1 □ ... □ K_mN
    hodge.py           # potential/harmonic/nonstrategic projection; alpha()
    generate.py        # synthesise families at target alpha
    potentialize.py    # nearest potential game (Candogan projection framework)
  solve/
    fixedpoint.py      # damped iteration, Anderson acceleration
    mirror.py          # magnetic mirror descent (last-iterate convergence)
    homotopy.py        # predictor-corrector arclength continuation
    implicit.py        # implicit-diff wrapper, shares the resolvent
    validate.py        # cross-check vs pygambit
  response/
    susceptibility.py  # chi_partial, chi_equilibrium (Result 1), matrix-free
    reciprocity.py     # reciprocity_defect (Result 2)
    spectral.py        # spectrum of SB, rho, critical lambda, bifurcation type
    fdt.py             # FDT identity checks, dispersion estimator
  dynamics/
    markov.py          # Glauber generator on profile space
    stationary.py      # pi* (sparse eigensolve / power iteration)
    currents.py        # J*, cycle decomposition
    entropy.py         # exact EPR, Hatano-Sasa, TUR bound
    estimators.py      # KLD k-th order, TUR-from-data, NEEP-style
    simulate.py        # trajectories, quench/anneal protocols
  demand/
    oracle.py          # PayoffOracle protocol  <-- key abstraction
    loglog.py  logit_demand.py  blp.py  dreamprice.py
  estimate/
    mle.py  ccp.py  dispersion.py  hierarchical.py  covariate.py  compare.py
  data/
    dominicks.py  panel.py  grid.py  markets.py
  viz/
    phase.py  branch.py  response.py  currents.py
  cli.py
```

### 8.3 The central abstraction

```python
from typing import Protocol
import jax.numpy as jnp


class PayoffOracle(Protocol):
    """Maps a joint price profile (+ optional state) to per-firm profit.

    Wraps ANY demand model. The library never assumes a functional form;
    it needs this map, and for gradient paths, JAX-differentiability.
    """

    n_firms: int

    def profit(self, prices: jnp.ndarray, state: jnp.ndarray | None = None) -> jnp.ndarray: ...
    def quantity(self, prices: jnp.ndarray, state: jnp.ndarray | None = None) -> jnp.ndarray: ...
    def elasticity(self, prices: jnp.ndarray, state: jnp.ndarray | None = None) -> jnp.ndarray:
        ...
        # (n_firms, n_firms) own + cross
```

`PricingGame` materialises the payoff tensor (or a matrix-free closure) by `vmap`-ing the oracle over the Cartesian product of per-firm grids.

**DreamPrice adapter.** DreamPrice is PyTorch; `thermoqre` is JAX. The causal demand decoder is `theta * log(price) + MLP(z_t, store_features)` — a log-price term plus a 3-layer MLP. **Port the weights to JAX; do not use `jax.pure_callback`** (it blocks JIT and grad). Keep a callback path behind a flag purely to validate the port; assert agreement to 1e-6 on a fixed batch in CI. Expose the frozen DML-PLIV elasticity (−0.940) directly as `elasticity`.

### 8.4 Key signatures

```python
# solve
def logit_qre(game, lam, *, method="mirror", init=None, tol=1e-10, max_iter=10_000) -> QREPoint: ...
def logit_branch(
    game, lam_max, *, n_points=200, first_step=0.03, max_accel=1.1, detect_bifurcations=True
) -> Branch: ...


# decompose
def normalise(game) -> Game: ...  # effective game; see §1.1
def hodge_decompose(game) -> tuple[Game, Game, Game]: ...
def alpha(game) -> float: ...
def make_family(potential, harmonic, alphas) -> list[Game]: ...


# response  -- the instruments
def chi_partial(qre_point) -> jnp.ndarray: ...
def chi_equilibrium(game, qre_point, *, matrix_free=False) -> jnp.ndarray:
    """(I - S B)^{-1} S  -- Result 1."""


def reciprocity_defect(game, qre_point) -> float:
    """||chi_eq - chi_eq^T||_F / ||chi_eq + chi_eq^T||_F  -- Result 2."""


def strategic_spectrum(game, qre_point) -> SpectrumInfo:
    """eigenvalues of S B; rho; distance to criticality; bifurcation type."""


def critical_lambda(game, *, bracket=(0.0, 1e3)) -> float: ...


# dynamics
def glauber_generator(game, lam): ...
def stationary(generator): ...
def probability_currents(generator, pi): ...
def entropy_production_rate(generator, pi) -> float: ...
def epr_from_trajectory(traj, *, method="kld", order=2) -> EPREstimate: ...
def tur_bound(traj, current_fn) -> float: ...


# estimate
def estimate_lambda_mle(game, counts, **kw) -> LambdaFit: ...
def estimate_lambda_ccp(panel, oracle, **kw) -> LambdaFit: ...
def estimate_lambda_dispersion(panel, oracle, **kw) -> LambdaFit: ...
def estimate_lambda_hierarchical(panel, oracle, groups, **kw) -> LambdaFit: ...
def compare_models(panel, oracle, models=("nash", "qre", "levelk", "qlk")) -> ComparisonTable: ...
```

`QREPoint` holds `sigma`, `lam`, `U`, `residual`, `n_iter`, and lazily-computed `chi_part`, `chi_eq`, `R`, `spectrum`.

### 8.5 Numerics — the things that will bite

- All softmax via `jax.nn.log_softmax` / `logsumexp`. Never exponentiate raw payoffs.
- Rescale payoffs to unit range internally, fold the scale into λ, and report both. Expose `lambda_normalised = lam * payoff_range`. λ is not scale-free and users will misread it.
- $C_i$ is rank-deficient by construction. **All linear algebra on $(I-SB)$ must happen on the tangent space** $T$, via an explicit orthonormal basis (Helmert, or QR of $I - \frac1m\mathbf{1}\mathbf{1}^\top$). Getting this wrong silently produces a spurious zero eigenvalue and a false criticality reading. Test the projection directly.
- Decompose on the **normalised** game (§1.1), not the raw tensor.
- Matrix-free: Lineax GMRES on $v\mapsto v - S(Bv)$, with $Bv$ a `vmap`-ed contraction, never materialising $B$.
- Implicit diff needs $(I-SB)^{-\top}$ — the same resolvent as the susceptibility. Implement once, reuse.
- Expose `distance_to_criticality = 1 - rho(SB)`; warn below $10^{-3}$ rather than silently returning a huge $\chi^{\text{eq}}$.
- Hodge: implement the separable Kronecker transform of §1.2, not a generic projection.

### 8.6 Tests

1. $\nabla\psi=\lambda\sigma$, $\nabla^2\psi=\lambda^2C$ to $10^{-12}$.
2. Potential game ⟹ QRE marginals match Gibbs marginals of $Z^{-1}e^{\lambda\Phi}$ to $10^{-10}$.
3. Potential game ⟹ $\mathcal{R}=0$, $J^*=0$, EPR $=0$, real spectrum of $SB$.
4. Adding a nonstrategic component leaves $\sigma^*$ invariant (K5 invariant).
5. `chi_equilibrium` matches finite differences of `logit_qre` w.r.t. $h$ to $10^{-6}$.
6. `logit_branch` matches `pygambit.qre.logit_solve_branch` on 2×2, 3×3 to $10^{-8}$.
7. `estimate_lambda_mle` matches `pygambit.qre.logit_estimate` on simulated counts.
8. Hodge decomposition: orthogonality, idempotence, and reconstruction to $10^{-10}$; separable transform matches a dense reference projection on small games.
9. λ→∞ principal branch → a Nash equilibrium (check via pygambit); λ→0 → centroid.
10. `epr_from_trajectory` recovers exact EPR on simulated Glauber trajectories from a known non-potential game, within CI.

---

## 9. API

FastAPI + Pydantic v2 + Uvicorn on Render (Docker). Async work via Redis-backed queue (`arq`) + a Render background worker. Postgres for jobs/metadata; object storage for uploads and artefacts. Anything over ~5 s goes async — branch continuation, dissipation estimation, and hierarchical λ are minutes-scale.

```
POST /v1/games/from-matrix        -> game_id
POST /v1/games/from-demand        -> game_id
GET  /v1/games/{id}

POST /v1/decompose                -> {alpha, norms, components_ref}
POST /v1/solve/qre                -> QREPoint            (sync, small)
POST /v1/solve/branch             -> job_id              (async)
POST /v1/response                 -> {chi_part, chi_eq, R, spectrum, rho,
                                      distance_to_criticality, bifurcation_type}
POST /v1/dynamics/stationary      -> {pi, currents, epr}
POST /v1/dynamics/epr-from-data   -> job_id

POST /v1/datasets                 -> dataset_id
GET  /v1/datasets/{id}/schema     -> mapping + validation report
POST /v1/estimate/demand          -> job_id
POST /v1/estimate/lambda          -> job_id
POST /v1/diagnose                 -> job_id   (full report)
POST /v1/optimize/price           -> {p_star, expected_profit,
                                      competitor_distribution, vs_nash, vs_myopic}

GET  /v1/jobs/{id}  |  GET /v1/jobs/{id}/stream (SSE)
GET  /v1/examples   |  GET /v1/health
```

Contract notes:
- Numeric payloads carry explicit `shape`/`dtype`; large arrays returned as signed URLs to Parquet/NPZ, never inline.
- Every response embeds `provenance` (library version, oracle, λ parameterisation, payoff normalisation constant) and `warnings` (`near_criticality`, `low_support`, `weak_instruments`, …).
- **`/v1/optimize/price` must return the competitor distribution alongside the point recommendation.** The whole economic argument is that the distribution over *rivals'* prices is the useful object, not a randomised own price. The UI shows both.
- API keys (`X-API-Key`) and rate limits from day one; free tier bounded by game size.
- Publish OpenAPI; generate a TS client for the frontend and a thin `thermoqre-client` Python package.

Render: `render.yaml` declaring `web` (Docker), `worker` (same image), `redis`, `postgres`. Health check `/v1/health`. Pin `jax[cpu]` in the deployed image and enforce size limits in validation. Warm the JIT on startup with a tiny game so the first request isn't 20 s.

---

## 10. App

Next.js (App Router) + TypeScript on Vercel. Tailwind + shadcn/ui. Charts: visx generally, Plotly for 3-D phase surfaces and heatmaps. KaTeX. TanStack Query against the API.

### 10.1 Learn

Interactive explainers with live controls, in order. These double as `docs/theory/` — write once, render twice.

1. **Softmax and λ** — payoff bars → probability bars, λ slider 0→∞, collapse to Nash.
2. **Fixed point** — animate $\sigma_B\to EU_A\to\sigma_A\to EU_B$ to convergence.
3. **QRE vs mixed Nash** — indifference vs. payoff-sensitivity; $EU(A)=10,EU(B)=8,P(B)>0$.
4. **MaxEnt** — slider on $1/\lambda$; maximising $\mathbb{E}[U]+\lambda^{-1}H$ *produces* the logit; show $F=U-TS$.
5. **Gibbs and potential games** — build a small congestion game, show $\Phi$, show QRE marginals coinciding with the Gibbs measure; then break potentiality and watch them separate.
6. **Detailed balance and currents** — animate the Glauber chain on the profile lattice; $J^*=0$ vs. circulating $J^*$. This is the visual heart of the project.
7. **Reciprocity** — poke firm 1, measure firm 2; poke firm 2, measure firm 1; watch the two numbers coincide at α=0 and diverge as α rises; $\mathcal{R}$ tracking α.
8. **Elasticity vs λ** — demand → payoff → game → QRE. Changing elasticity reshapes the payoff surface; λ only changes response sharpness. Forestalls the "QRE implicitly models demand" confusion.
9. **The one-price objection** — reproduce the PI's own critique honestly: if you just want tomorrow's price, argmax expected profit. Then show that the expected profit is computed against the *competitor distribution*, which is what QRE supplies. This section is the app's intellectual credibility.

### 10.2 Lab

Stream B's actual instrument. Build it early; the PI should be using it daily.

- Game builder: templates (Bertrand duopoly, capacity-constrained, coordination, congestion) or upload a payoff tensor.
- Two master sliders: **λ** and **α**. Everything below updates live.
- Panels: per-firm price distributions; QRE branch with turning points marked; spectrum of $SB$ in the complex plane with the unit circle drawn (read the bifurcation type straight off it); $\mathcal{R}$ gauge; probability-current field on the profile lattice; dissipation readout; running $(\lambda,\alpha)$ phase heatmap with the current position pinned.
- "Anneal" button: λ-quench with the Jarzynski/Crooks work distribution. This is the demo that makes the thermodynamics visceral.
- Export: every panel to PNG/SVG, underlying arrays to CSV, and a **"reproduce this in Python"** button emitting a runnable `thermoqre` snippet. That last one is the main funnel from app to library.

### 10.3 Analyze

1. **Upload** a panel (CSV/Parquet) or use the bundled Dominick's example. Minimum schema: entity id, time, price, quantity. Optional: cost, promotion flag, competitor id/price, market id, covariates.
2. **Schema mapping** with inference and a validation report — missingness, price-grid density, panel balance, whether there's enough cross-sectional variation for the chosen estimator. Refuse loudly and *specifically* when the data can't identify what's being asked.
3. **Demand stage.** Oracle choice: constant-elasticity (always available), logit/nested logit, BLP (if instruments supplied), or a custom/uploaded model. Report elasticities with CIs and first-stage strength.
4. **Strategic stage.** λ from all four estimators shown together with agreement flagged; $\mathcal{R}$; TUR dissipation bound; phase-map location of this market.
5. **Decision stage.** Competitor price distribution; expected-profit curve over the grid; $p^*$; comparison against Nash-Bertrand, cost-plus, match-competitor; out-of-sample backtest if history allows.
6. **Report.** Downloadable HTML/PDF with the full analysis, assumptions, warnings, and a plain-language interpretation section.

The report states plainly that λ absorbs unmodelled heterogeneity, that low λ is not proof of irrationality, that recommendations are conditional on the demand model, and that this is decision support rather than an autonomous pricing system. Carry over DreamPrice's "NOT intended for" list. Say it once, clearly.

### 10.4 Vercel

Static/ISR for Learn; client-side fetching for Lab and Analyze. Heavy compute always goes to the Render API — Vercel functions only proxy auth and short requests. Signed direct-to-storage uploads; Vercel never holds the file. Optional later: a tiny WASM/TS logit solver so the Learn sliders are instant and work offline (do not attempt for Lab or Analyze).

---

## 11. Build order

Sequential within Stream A because of dependencies; streams B/C/D overlap freely.

**Stream A (library core)**
1. Scaffold, `pyproject.toml`, CI, float64, `types.py`.
2. `games/tensor.py`, `games/normalise.py`, `games/library.py` — including a verified exact potential game and a pure harmonic game (RPS).
3. `solve/fixedpoint.py`, `solve/mirror.py`. Tests 1, 9.
4. `solve/validate.py` vs pygambit; `solve/homotopy.py`. Tests 6, 9.
5. `response/susceptibility.py` with tangent-space projection done right. Test 5.
6. `response/reciprocity.py`, `response/spectral.py`. Test 3.
7. `decompose/kron.py` → `hodge.py` → `generate.py`. Tests 4, 8.
8. `dynamics/*` — generator, stationary, currents, exact EPR. Tests 2, 3, 10.
9. `solve/implicit.py` reusing the resolvent.

**Stream C (data, parallel from step 5)**
10. `demand/oracle.py`; `demand/loglog.py`; `games/factored.py`.
11. `demand/dreamprice.py` — port the decoder, validate against torch.
12. `data/*` — Dominick's loader, grids, markets.
13. `estimate/mle.py` → `ccp.py` → `dispersion.py` → `hierarchical.py` → `compare.py`. Test 7.

**Stream D (product, parallel from step 6)**
14. FastAPI app, Pydantic schemas, sync endpoints.
15. Queue + worker + Postgres; async; SSE.
16. `render.yaml`, Dockerfile, JIT warm-up, limits, API keys.
17. Next.js scaffold, design system, chart primitives.
18. Learn (nine explainers).
19. Lab.
20. Analyze.

**Stream B** runs continuously from the moment step 7 lands, mostly through the Lab UI once step 19 exists.

---

## 12. Repository layout

```
thermoqre/
  packages/
    thermoqre/                  # library (PyPI)
    thermoqre-client/           # thin API client
  services/api/                 # FastAPI + worker; Dockerfile; render.yaml
  apps/web/                     # Next.js; vercel.json
  experiments/
    calibration/                # potential-game instrument checks
    alpha_sweep/                # the phase map
    dominicks/                  # data contact
    Makefile                    # `make reproduce` — fixed seeds
  papers/
  docs/                         # mkdocs-material; docs/theory/ = Learn source
  CLAUDE.md                     # agent notes, conventions, gotchas
```

---

## 13. Things most likely to go wrong

| Risk | Level | What to do |
|---|---|---|
| Tangent-space projection bug → phantom criticality | **High** | Dedicated tests; refuse to report χ near singularity; cross-check vs finite differences |
| Decomposing the raw rather than normalised game (§1.1) | **High** | `normalise()` is mandatory in the decompose path; test that strategically equivalent games give identical α |
| λ not separable from collusion/misspecification | **High** | Report as bounded; lead with the λ-free reciprocity angle |
| Dominick's rival-firm interpretation is weak (single chain, zone pricing, 1989–97) | **High** | Treat as an illustrative anchor; design Analyze so users bring genuinely competitive data |
| arXiv:2405.07224 already covers the α-sweep dynamics | Medium | Read it first; differentiate on the thermodynamic observables (currents, EPR, TUR), or collaborate |
| DreamPrice torch→JAX port drift | Medium | Keep the callback path; CI assertion to 1e-6 on a fixed batch |
| EPR estimators unreliable under partial observation | Medium | TUR *lower bound* as headline; validate on synthetic ground truth first |
| "Physics metaphor" pushback | Medium | The §3 confidence labels are the defence. Exact identities are exact; Results 1–3 are proved or checked; nothing decorative enters a claim |
| Scope sprawl across library + API + app | Medium | Streams are parallel but each is independently useful; Learn ships before Lab, Lab before Analyze |
| Hodge decomposition scaling | ~~High~~ **Low** | Separable Kronecker transform (§1.2) — near-linear, no iterative solver |
| Render cost for JAX workloads | Low | CPU image, enforced size limits, async queue, cache on (game hash, λ) |

---

## 14. Do these next

1. **Read arXiv:2405.07224** end to end. Nearest live work; determines whether Stream B is differentiation or duplication.
2. **Scaffold the library and get `reciprocity_defect()` returning 0 on a congestion game and >0 on RPS.** First genuinely new artefact.
3. **Implement the separable Kronecker Hodge transform.** Self-contained, satisfying, unblocks Stream B.
4. **Port the DreamPrice causal decoder to JAX.** Independent of everything else; unblocks Stream C.
5. **Write `docs/theory/01`–`09`** as the shared source for docs and Learn mode.

Then go slide the sliders and see what's there.
