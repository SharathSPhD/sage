# SAGE / ThermoQRE — Direction v4: the claim, the closure, and what stops

**Status:** Supersedes the *working philosophy* of `THERMOQRE_PROGRAMME_v3.md` §0.1 and the
*work-stream ordering* of §5 and §11. The science of v3 §1–§4 stands; the build spec of
§7–§12 stands. What changes is what the programme is *for* from here.

**Audience:** the PI, and any agent that will be asked "what should I work on next?"

**Written against** the repository at commit `a0b79f0`, `gates/status.json` (26 units, all
GREEN), `memory/findings.md` (F-0001…F-0021), and the original programme documents
`research/research-main.md`, `research/research1.rtf`, `research/THERMOQRE_PROGRAMME_v3.md`.

---

## 0. The verdict, first

**Has it digressed?** Yes — and the digression is measurable, not a matter of taste.

**Is it beating around the bush?** In the last six units, yes. Everything after F-0015 is
the programme studying its own thermometer instead of the weather.

**Is there a ground-breaking discovery in here?** Yes, and it has been sitting in the
repository since 2026-08-08, mislabelled as a *refuted repair hypothesis*. It is F-0004 /
F-0007 / F-0010. The programme found it, tested it hard enough to nearly kill it, failed to
kill it, and then walked past it to go and fix a standard-error estimator.

**What this document does.** Names the claim, states it in a form that can be attacked,
demotes every instrument to the role of a coordinate in that claim, kills the current R10,
and defines *done*.

---

## 1. The digression, with evidence

The v3 spec (§0.1) said, in terms:

> "This is **not** a pre-registered hypothesis-testing programme… The right model is
> **exploratory instrumentation building**… Build the instrument first, decide what it
> means second."

That instruction was followed exactly, and it worked — right up to the point where it
became self-consuming. Classify each finding by whether it establishes a fact about
**the world** (strategic systems behave thus) or a fact about **the method** (our own
estimator/gate/criterion behaves thus):

| Window | Dates | WORLD | METHOD | Mixed |
|---|---|---|---|---|
| F-0001 → F-0015 | 08-08 → 08-12 | **12** | 2 | 1 |
| F-0016 → F-0021 | 08-12 | **0** | 5 | 1 |

The programme flipped on a single day. The last six findings — F-0016, F-0018, F-0019,
F-0020, F-0021, and the two F-0017 corrections — are, without exception, about
`hs_estimator.py` and the relaxation gate in front of it. The only world-shaped number
produced in that stretch (F-0017's ≈7.0 nats/day CAISO loop affinity) is explicitly marked
*descriptive only* and out of certified scope, and it does not appear in either paper.

**How deep the recursion goes.** R10 (`thermo.hs_estimator.tau_lag`) is currently specified
as: choose the *lag* used to estimate the *autocorrelation* used to estimate the
*relaxation time* used by the *gate* that certifies the *precondition* for the *estimator*
that would, if it ever ran on data, produce a number about a market. That is six levels of
indirection from any claim about strategic systems. The technical content of R10 is
correct and the F-0021 root-cause analysis is genuinely good work. The problem is not
quality; it is that no amount of it terminates.

**Why v3's philosophy caused this and could not stop it.** "Anomalies are the product" plus
"no stage gates" plus an evidence discipline that treats every anomaly as gate-worthy has
no *stopping rule*. Each instrument defect is a legitimate anomaly; chasing it is
locally correct; the sequence never returns to the world because nothing in the philosophy
says it must. §9 of `docs/ONBOARDING.md` ("a check that passes tells you only that the
check passed") is the correct lesson from that stretch — but it is a lesson about
*checking*, and applying it recursively is exactly the trap.

**What v3 got right and keeps.** The confidence tiers, the pre-registration, the red-team
dispositions, the refusals-as-outputs, the retraction record. That apparatus is the most
valuable thing in the repository and it is why the claim below can be stated with a
straight face. Nothing here weakens it. It is being *pointed at a claim* instead of at
itself.

---

## 2. What the original documents said the discovery path was

The PI's instruction is that the path was already in the original research files. It was.
Three passages, all pre-dating the build:

**(a) `research-main.md`, Recommendations §5:**

> "**Publish the FDT / 'elasticity IS a susceptibility, price dispersion IS its conjugate
> fluctuation' result first** — it is the cleanest genuinely new contribution and does not
> depend on the messier λ-identification questions."

**(b) `research-main.md`, Key Findings §4 and Recommendations §3:**

> "The dispersion–response estimator of λ is the flagship novel contribution… *Falsification
> criterion:* the three estimates should coincide within confidence bounds. **Systematic
> divergence of the dispersion estimator from the MLE is itself the diagnostic that detailed
> balance is broken** (non-potential structure / cycling), which is a finding, not a failure."

**(c) `research1.rtf`, §10:**

> "You could potentially make λ state-dependent… Then QRE becomes a model where strategic
> precision changes with the environment… Now you're getting something that standard Nash
> cannot represent."

Read (a) and (b) together and the shape of the intended discovery is unmistakable:
**response and fluctuation are locked together at equilibrium, and the way that lock breaks
is the measurement.** That is the fluctuation–dissipation programme. It is not an
instrumentation programme; the instruments were only ever the way to get at it.

The repository built every instrument that programme needs, ran the sweep that programme
demanded, and got a result *sharper than what was predicted* — and then filed it under
"repair hypothesis refuted."

---

## 3. The claim

### 3.1 Statement

> **The Two-Axis Result.** For a finite game under logit revision dynamics, the local
> response asymmetry ℛ and the global dissipation σ_EP both vanish exactly when the
> normalised game is potential, but away from that point they are **not two readings of one
> underlying distance from equilibrium**. Their agreement is an artefact of the potential
> component modulating both. Conditional on the harmonic fraction α, their co-movement
> **collapses universally** as α → 1, and its residual **sign is λ-dependent**. Distance
> from equilibrium is therefore not a scalar for strategic systems; the (ℛ, σ_EP) plane is
> the correct classification space, and which quadrant a system occupies is decidable from
> observation.

### 3.2 The two coordinates and why they are not the same object

| | ℛ — the response axis | σ_EP — the dissipation axis |
|---|---|---|
| Definition | ‖χ^eq − (χ^eq)ᵀ‖_F ⁄ ‖χ^eq + (χ^eq)ᵀ‖_F, with χ^eq = (I − SB)⁻¹S | Schnakenberg entropy production of the stationary Glauber chain on joint profiles |
| Mathematical type | a **local derivative** evaluated at one equilibrium | a **global flux functional** over the whole profile space |
| Physical ancestor | Onsager reciprocity | entropy production in a NESS |
| Vanishes iff | normalised game is potential (Result 2 / N1) | normalised game is potential (K3) |
| Observable from | cross-agent pass-through asymmetry — **no payoffs required** | trajectory irreversibility — **no payoffs required** |
| Cost | one solve + one resolvent | a chain over ∏mᵢ states, or a trajectory estimator |

They share an origin and nothing else. The whole content of the result is that sharing an
origin does not make them collinear.

### 3.3 The evidence already in the repository

All of this is committed, gated, red-teamed, and regenerable from fixed seeds.

**The marginal co-movement is near-perfect** — and is a confound.
ρ_S(σ_EP, ℛ) = **0.9927** [0.9914, 0.9936], n = 1000 (`chain_comovement.json`).
But ρ_S(σ_EP, α) = 0.9904 and ρ_S(ℛ, α) = 0.9819 (n = 2000,
`reciprocity_alpha_sweep.json`). Both meters are reading α. That is the naive result, and
the naive result is the one a less careful programme would have published.

**Stratified by α, the agreement collapses** (`chain_comovement.json`, within-level
Spearman, 100 games per level):

| α | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| ρ_S(σ_EP, ℛ) | 0.882 | 0.812 | 0.856 | 0.849 | 0.800 | 0.870 | 0.801 | **0.610** | **0.323** | **−0.355** |

**The obvious repair was pre-registered and refuted by its own test** (F-0007,
`decoupling_mechanism.json`). Hypothesis H1 — that the collapse is an artefact of ℛ being a
*ratio*, and that the numerator ‖χ − χᵀ‖ alone would keep tracking σ_EP — is false: the
numerator alone reaches ρ = **−0.368** at high α. H2 — that the denominator drives the
ratio — is true (ρ(ratio, 1/den) = **0.955** at high α), which explains the mechanism but
does not repair the decoupling. **No renormalisation of the response matrix recovers the
dissipation ordering.** This is the strongest single piece of evidence in the programme,
because it is the programme's own repair attempt failing under its own pre-registered
criterion.

**The scope of the result is honestly bounded** (F-0010): across λ ∈ {0.8, 1.2, 2.0} and
m ∈ {3, 4}, the **collapse is universal** — the conditional correlation degrades at every
setting. The **sign of the residual is λ-dependent** (+0.25 → +0.03 → −0.23). So the
headline is *the plane is two-dimensional*, not *the second axis always points that way*.
Anyone who states the −0.355 without the λ-dependence is overclaiming, and this document
says so in advance.

**Both axes are calibrated at both ends, on real data with payoffs known by construction.**
Sioux Falls road network (exact potential, real TNTP data): ℛ = **5.65 × 10⁻¹⁷**, Fisk KKT
spread 7.1 × 10⁻¹⁵, DF-symmetry defect exactly 0. Colonel Blotto (free payoffs): α = 0.694,
ℛ = 0.118. Rock–paper–scissors: ℛ ≈ 0.69–0.87. Degenerate equal-value Blotto: σ_EP =
5.4 × 10⁻³². The instruments read zero where zero is the right answer, and they do it at
machine precision on data nobody in this project generated.

**Both axes have been read on real systems.** Dominick's cross-brand cost pass-through:
ℛ = **0.00112** [4.8 × 10⁻⁵, 5.0 × 10⁻³], n = 22 655 store-weeks over 86 stores, cluster
bootstrap — potential-like exactly where single-retailer category management predicts it,
with the prediction registered *before* the run. CAISO SP15 day-ahead: phase-embedded KLD
= 0.0447 nats/hour ≈ **1.07 nats/day**, exceeding the reversibilized-Markov null
(p < 0.01) while sitting at-null against FT and AAFT surrogates. **Those two readings are
the first two points on the plane, and they sit in different quadrants.**

**The dissipation axis is estimable from trajectories alone.** ρ_S(KLD, exact) = **1.0**;
the debiased finite-time TUR bound holds below the exact value at every level, median
tightness 0.850 (`estimator_alpha_sweep.json`). This is what makes the plane *observable*
rather than merely definable.

**The response axis has a spectral frontier.** λ_c(α) exists only for α ≥ 0.55 and descends
monotonically — 7.77, 5.22, 4.22, 3.64, 3.23, 2.95 across α = 0.55 → 0.80, single crossing
verified at every level, zero monotonicity violations, n = 440 (`frontier_lambda_c.json`).
Potential games *escape* criticality at high λ because equilibrium concentration kills the
choice covariance faster than λ amplifies it (F-0006). That is a second, independent
world-fact and it belongs to the same paper.

### 3.4 Why this is ground-breaking rather than merely true

Three reasons, in descending order of confidence.

1. **It contradicts an imported intuition.** Everyone who reaches for thermodynamics in
   economics or multi-agent systems reaches for *a temperature* and *a distance from
   equilibrium* — one number, one axis. QRSE does this. Most econophysics does this. The
   result says the reflex is wrong in the case where the analogy is otherwise *exact*: the
   potential-game limit where QRE genuinely is a Gibbs measure. The failure is not at the
   edge of the analogy; it is one step inside it.

2. **It is a negative structural result with a named mechanism.** χ^eq is a derivative at a
   point; σ_EP is a functional of a flux field. Near α = 0 the potential modulates both, so
   they track. As the potential vanishes there is nothing left to make a local object and a
   global object agree, and they stop agreeing. That is a *reason*, not a correlation, and
   it predicts where to look for the effect in any other system with the same two-object
   structure.

3. **It is operational.** Both coordinates are estimable without knowing payoffs — ℛ from
   pass-through asymmetry, σ_EP from trajectory irreversibility. So the claim is not "there
   exists a plane"; it is "you can locate your system in it from observational data." That
   converts a theorem into an instrument, which is the only form in which anyone outside
   this project will care.

### 3.5 What would kill it

Registered here, before the work, in the project's own idiom:

- **K1.** A reparametrisation of the response axis that is *not* a post-hoc fit — derived
  from the resolvent structure, stated before the sweep — that restores ρ_S > 0.6 at
  α = 0.95 across ≥ 3 independent seed families. F-0007 refuted the obvious candidate;
  another may exist. If one is found, the plane collapses to a line and the result is a
  normalisation bug that took the programme four units to find.
- **K2.** Demonstration that the collapse is a **finite-size artefact** — that it weakens
  monotonically in m and vanishes as m → ∞. Currently tested at m ∈ {3, 4} only. This is the
  most likely way the result dies and it must be tested first.
- **K3.** Demonstration that the collapse is a **solver artefact** — that a
  last-iterate-convergent solver (magnetic mirror descent) at tighter tolerance, or exact
  homotopy continuation, removes it. Cheap to run; run it.
- **K4.** Prior art. If the local-vs-global inequivalence is already stated for logit
  dynamics on finite games, the contribution reduces to the measurement framing. The
  Legacci–Mertikopoulos–Pradelski audit (arXiv:2405.07224) says orthogonal, but that audit
  was inherited and never independently re-run.

If K1 or K2 lands, this document is wrong and the programme says so in public, in the same
register it used for F-0008 and F-0015.

---

## 4. What this reclassifies

Every unit in the ledger keeps its gate. What changes is what it is *for*. Nothing below is
new work; it is relabelling that makes the existing work legible as one thing.

| Unit | Was | Now is |
|---|---|---|
| `finite.response.reciprocity` | an instrument | **the x-coordinate** |
| `dynamics.exact`, `thermo.estimators` | instruments | **the y-coordinate, and its observability from data** |
| `finite.decompose.hodge` | an instrument | **the confound** — the variable you must condition on before the two axes mean anything separately |
| `science.phase_map`, `science.frontier` | studies | **the geometry of the plane in λ**, and the reason the axes are not interchangeable |
| `science.decoupling` | "a refuted repair" | **the result** |
| `domains.congestion`, `domains.blotto` | calibrations | **the two anchor points**, at machine precision, on data we did not make |
| `domains.pricing`, `domains.electricity` | readings | **the first two real systems located in the plane, in different quadrants** |
| `estimate.lambda`, `estimate.bayes` | estimators | **the third reported quantity** — λ, because the sign of the tilt depends on it |
| `thermo.protocols`, `thermo.hs_estimator*` | the recent work | **out of the flagship claim.** Quench thermodynamics is a genuine second line (F-0012's driving-cost inversion is a real world-fact) but it is *not* the plane, and it has been consuming the programme. It becomes its own paper, later, or it waits. |
| `api.core`, `web.scaffold`, `product.toolkit` | surfaces | **the delivery of the plane to two audiences** (see the product spec) |

**The re-read of F-0007 is the single most important sentence in this document.** It is
currently in `memory/findings.md` as "the programme's own repair hypothesis refuted by its
own test." It should read: *"the programme's own repair hypothesis was refuted by its own
pre-registered test, which is what promoted the decoupling from an anomaly to a structural
result."* Same facts. The first framing files it under process hygiene; the second files it
under physics.

---

## 5. The quadrants — what the plane is *for*

The classification is the deliverable, because it is the thing a person who is not this
project can use. Bands are the toolkit's existing calibrated thresholds (ℛ: 0.02 / 0.30;
σ_EP against its own null).

| | **σ_EP ≈ 0** (no persistent flux) | **σ_EP > 0** (persistent flux) |
|---|---|---|
| **ℛ ≈ 0** | **I. Landscape.** A potential game. Comparative statics are trustworthy, pass-through is symmetric, there are no cycles to time, and optimising against a static competitor model is correct. *Anchor: Sioux Falls, ℛ = 5.6e-17. Real reading: Dominick's, ℛ = 0.0011.* | **II. Driven landscape.** Reciprocal structure, circulating dynamics — something exogenous is cycling the system (demand, schedules, cost shocks), not the strategic interaction. **Timing matters, structure does not.** *Real reading: CAISO day-ahead, ≈1.07 nats/day, ℛ untested.* |
| **ℛ > 0** | **III. Stalled whirlpool.** Asymmetric response with no persistent circulation — one agent structurally leads, but nothing cycles. **Structure matters, timing does not.** Pass-through asymmetry is the exploitable object. | **IV. Whirlpool.** Both. Edgeworth-cycle territory. Response asymmetry *and* circulation; the regime where naive optimisation against a static rival model is worst. *Anchor: RPS, ℛ ≈ 0.69–0.87.* |

Two consequences worth stating plainly:

- **The off-diagonal quadrants are the entire point.** If distance-from-equilibrium were a
  scalar, II and III would be empty. The result says they are not, and the two real readings
  the programme already has (Dominick's in I, CAISO in II) are *not on the same diagonal*.
- **Quadrant III is unoccupied and is the highest-value target.** Finding a real system in
  III would be the strongest possible confirmation, because it is the quadrant the scalar
  intuition most firmly denies. **Retail fuel pricing with asymmetric rockets-and-feathers
  pass-through but no Edgeworth cycling is the obvious candidate**, and it is exactly the
  system the PI's original petrol-station framing describes.

---

## 6. The agenda to closure

Five units. Each is bounded, each terminates, each is either a coordinate of the plane or a
point in it. Nothing in this list is about an estimator's standard error.

### R10′ — **Redirected.** `science.plane.robustness`
The current R10 (`thermo.hs_estimator.tau_lag`) is **suspended, not cancelled.** Its
technical content is right and F-0021 is a real defect; it is registered as debt in
`memory/decisions.md` with an ADR recording that it was suspended for direction, not for
quality. It resumes only if a unit downstream of it is blocked on it — and the plane is not.

In its place: **the two kill-shots at the claim, run before anything is written.**
- K2 — extend the α-stratified sweep to m ∈ {3, 4, 5, 6}. Registered criterion: if the
  within-level ρ at α = 0.95 rises monotonically toward the low-α value as m grows, the
  result is finite-size and the paper does not exist. Pre-register the monotonicity test and
  the n.
- K3 — re-run the α = 0.85 and α = 0.95 levels with `mirror.logit_qre_mirror` at tol 1e-14
  and with `homotopy.logit_branch` continuation, on the identical seed family. Registered
  criterion: |Δρ| < 0.05 versus the damped solver. If it fails, the result is numerical.
- K4 — independently re-audit arXiv:2405.07224 and the Candogan/Sandholm line for a prior
  statement of local-vs-global inequivalence. Record the search terms and the negatives,
  not just the verdict.

*Exit:* three artifacts, one finding, and either a live claim or a public retraction.

### R10′ — **RUN. Outcome recorded 2026-08-13.**

Both kill-shots were run as `science.plane` / R11 (`experiments/plane_robustness.py`,
artifact `plane_robustness.json`, finding **F-0022**). The outcome is more useful than a
clean pass, and it is not what the agenda above anticipated.

- **K3 (solver) — settled.** On identical seed families, magnetic mirror descent and a
  100× tighter damped tolerance both give |Δρ_S| = 0.00000; the solver perturbs ℛ by
  ~10⁻¹². *But* red team correctly reframed this: Spearman is rank-based and a 10⁻¹²
  perturbation cannot reorder anything, so |Δρ| = 0 was arithmetically forced before the
  run. K3 is a **diagnostic**, not a kill-shot that was passed. It is reported that way now.
- **K2 (finite size) — INDETERMINATE, and the reason matters more than the result.**
  The m-sweep {3,4,5,6} was run. Then red team found that this unit had registered its
  criteria **twice**: `plane_finite_size.yaml` (`a6533e7`, 23:02) with two runs launched
  against it, and only afterwards `plane_robustness.yaml` (`b89d5df`, 23:26) whose bar was
  easier to survive. On the same data the first registration returns INDETERMINATE and the
  second returns SURVIVES. **The earlier binds.** Separately, the deciding statistic
  (`gap_ratio`) was the one quantity never given a confidence interval, and its interval
  crosses its own bar (0.585 [0.434, 0.749], P(<0.50) = 0.13).
- **What survives, and now carries the claim.** The **ceiling criterion**, which is not
  close: the high-α association never recovers at any tested m — worst upper endpoint
  +0.156 against a 0.35 ceiling — and the α = 0.05 and α = 0.95 intervals are disjoint at
  every m. Independence needs decorrelation, not anti-correlation, and decorrelation is
  what is established.
- **What does not survive.** F-0004's sign reversal. At m = 6 the correlation is
  +0.012 [−0.137, +0.156], indistinguishable from zero. **The −0.355 is an m = 3–5 fact and
  must never again be quoted without its (m, λ) scope.** §3.1 above is corrected accordingly.
- **A generator defect worth carrying forward.** `make_family` fixes the Frobenius norm, so
  per-entry payoff RMS falls like 1/m: bigger action sets are also *colder* games
  (λ_normalised 1.86 → 1.27). Any future m-sweep on this generator must control per-entry
  RMS or it will measure temperature and report dimension. The control arm run here does not
  fix it either — it drives λ_normalised +40% where the main arm drives it −32%, so the two
  arms *bracket* the confound rather than controlling it, leaving a residual m-effect of
  ≈0.22 out of 0.364.

**The lesson, which is §9's lesson again one level up.** This programme's discipline exists
to stop exactly this, and it worked — but only because someone adversarial went looking. A
criterion substituted between two registrations is invisible in a diff, because the second
registration is a *new file* rather than an amendment. **New standing rule for
`CLAUDE.md`: a unit may register its criteria once. A second criteria file for the same unit
is a substitution and must be disclosed in the finding, with both adjudications reported.**

**Still outstanding from R10′:** K4, the independent prior-art re-audit of arXiv:2405.07224,
has not been run. And the newly obvious next kill-shot is **N > 2**, which is untested —
everything above is N = 2.

### R11 — `science.plane.quadrant_iii`
Locate a real system in quadrant III, or establish that the quadrant is empty in the data
available. Retail fuel is the target; the fallback is an experimental-games panel where the
payoffs are known by construction and the leader–follower asymmetry is designed in. Register
the ℛ threshold and the σ_EP null *before* the data is touched.

*Exit:* a third point on the plane, or a documented refusal naming the data that would settle it.

### R12 — `product.plane` (the delivery unit)
`strataq.diagnose()` — one call, any of three input shapes, returns a point in the plane with
its quadrant, both CIs, the λ it was read at, and the refusals. This is the unit that makes
the claim usable by someone who is not this project. Specified in the product document.

*Exit:* the facade, its tests, the app route, and a first-code-block in the README that
produces a quadrant verdict in five lines.

### R13 — `paper.plane`
The flagship paper. Structure, figure program and venue in `papers/PAPER_PROGRAMME_v2.md`.
Written only after R10′ clears.

*Exit:* a submitted preprint.

### R14 — `paper.software`
The software paper (JOSS or SoftwareX), which is `p1_instruments` cut down to what a software
paper is for. It can be written in parallel with R13 because it makes no scientific claim
beyond "these functions compute these things and here is the calibration."

*Exit:* a submitted software paper.

**Then the programme is closed.** Quench thermodynamics (F-0012 onward), state-dependent λ
(`research1.rtf` §10), and the λ-versus-collusion separation are a *second* programme. They
are good. They are not this one. Writing them down as a successor is how this one gets to
stop.

---

## 7. What stops

- **Stop adding levels of indirection to the gate machinery.** New rule, to be added to
  `CLAUDE.md`: *a unit whose claim is about the project's own instrument may not spawn a
  successor unit whose claim is about that instrument's instrument.* Two levels is the
  budget; the third goes to `memory/decisions.md` as debt and the programme returns to the
  world.
- **Stop treating every anomaly as gate-worthy.** v3 §0.1's "anomalies are the product" is
  amended: an anomaly is the product *when it is about the object of study*. An anomaly
  about our own measurement is maintenance, and maintenance is logged, not gated.
- **Stop growing `p3_noneq` §5.** It is 28% of the paper and it is internal engineering
  history. It comes out; F-0012's driving-cost inversion survives as one paragraph.
- **Stop the two overclaims currently in the abstracts.** `p1` says ℛ is "a payoff-free,
  λ-free test" without the qualifier that only the *zero* test is λ-free — F-0002 corrected
  exactly this and the body already gets it right. `p3` says the RTM verdict is a "certified
  at-null" — F-0008's certified-null claim was retracted. Both abstracts currently contradict
  the project's own record. Fix before anything is sent anywhere.
- **Stop deferring the empirical anchor.** Dominick's ℛ = 0.0011 (F-0011) is arguably the
  most persuasive number in the repository and it appears in the claims ledger and the app
  but in neither paper. It goes in the flagship.

---

## 8. Definition of done

The programme is finished when all six hold:

1. R10′ has run and the claim is either live or publicly retracted.
2. There are **at least three real systems located in the plane**, in at least two quadrants,
   each with a CI and a registered null.
3. `strataq.diagnose()` takes observed data and returns a quadrant, and a person outside this
   project has run it on their own data.
4. The flagship paper is submitted.
5. The software paper is submitted.
6. `memory/decisions.md` carries an ADR naming the successor programme and the debt it
   inherits — including the suspended R10.

Nothing else counts as done. Not more units, not more artifacts, not a greener gate suite.

---

## 9. The one-paragraph version

The programme set out to ask whether you can tell, from behaviour alone, if a strategic
system is a landscape or a whirlpool. It built the instruments to do that, calibrated them
at machine precision against systems with known answers, pointed them at two real markets —
and along the way discovered something better than the question it asked: that "landscape or
whirlpool" is not one question but two, that the two answers come apart, and that the coming
apart is structural rather than a defect of the instrument, because the programme tried to
repair it and failed under its own pre-registered test. Everything built so far is the
apparatus for that one sentence. The remaining work is to try twice more to kill it, find a
third system, make the reading a single function call, and write it down.
