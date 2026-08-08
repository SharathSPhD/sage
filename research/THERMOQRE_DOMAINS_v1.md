# ThermoQRE — Domain Expansion (companion to PROGRAMME v3)

**Status:** Companion document. Read alongside `THERMOQRE_PROGRAMME_v3.md`, which remains the master spec. This doc covers *which domains beyond pricing to support, in what order, and at what real cost.*
**Ethos:** unchanged from v3 §0.1 — build instruments, point them at things, chase anomalies. Nothing here is a hypothesis to adjudicate.

---

## 1. Verdict on the expansion proposal

**The core recommendation is right and should be adopted:** build the library around a generic game object, make pricing the flagship application rather than the definition. The argument in the proposal's §13 is the strongest one and it is correct — a fluctuation–response relation demonstrated only on grocery scanner data is weak evidence about anything; the same relation demonstrated across a potential congestion game, a non-potential oligopoly, an auction and a Blotto game is evidence about *structure*.

**But the proposal's domain ranking is derived from the wrong criterion.** It ranks on data availability and practical value. Those matter for a product; they don't tell you whether the *instruments* in v3 §3 will work. The instruments need something more specific, set out in §2 below, and applying that criterion substantially reorders the list.

**Four specific corrections**, each verified:

1. **Traffic is badly under-rated and mis-specified.** The proposal suggests NGSIM. NGSIM is vehicle-trajectory (car-following) data at the wrong granularity for route choice, and it has well-documented systematic errors — a critical re-extraction study found the raw data exhibits trends absent from manually re-extracted trajectories, with errors depending on speed, location and vehicle length, and other work reports errors that cannot be fixed by cleaning or interpolation. The right resource is the **TNTP benchmark set** (`github.com/bstabler/TransportationNetworks`): standard networks with OD matrices, BPR link cost functions, and best-known equilibrium flows at ~1e-15 average excess cost. More importantly, congestion games with BPR costs are **exact potential games** (Rosenthal 1973; Beckmann transformation), and **logit route choice is Fisk's (1980) stochastic user equilibrium — which is literally entropy-regularised Wardrop equilibrium with a known convex potential.** That makes traffic the single best calibration domain in the entire programme: it is the only place where the potential is known analytically, the payoffs need no estimation, and there are 45 years of estimated dispersion parameters θ (= λ) to check against. It should move to Tier 1 and arguably be built *first*.

2. **Electricity data: ENTSO-E is the wrong source; ERCOT is the right one.** ENTSO-E Transparency publishes generation, load, transmission and balancing — aggregate quantities, not firm-level offer curves. What the instruments need is unit-level bidding behaviour. **ERCOT's 60-day disclosure reports** publish exactly that: `60d_DAM_EnergyOnlyOffers`, `60d_SCED_Gen_Resource_Data`, energy offer curves with resource identity, free and public with a 60-day lag. There is also a newer "Energy Offer Curve Updates in Operating Hour" report. Nord Pool publishes aggregate bid/ask curves shortly after settlement — useful, but aggregated, so weaker for firm-level strategic inference.

3. **Auctions are not a plugin — they are a third engine.** Private values mean type spaces, which means Bayesian games, which means the entire finite-strategic-form machinery in v3 (the $S$, $B$, resolvent, Hodge decomposition) does not apply as written. This is a substantial cost the proposal doesn't surface. Auctions remain worth doing — Goeree, Holt & Palfrey's QRE explanation of overbidding in private-value auctions (*JET* 2002) is canonical and the domain is conceptually the cleanest — but budget for it as core work, not as a `games/auction.py` file.

4. **Sports should be demoted from a research domain to a teaching domain.** The proposal is right that penalty kicks make QRE intuitive, and it is the best answer to the PI's own "but a firm chooses one price" objection. But there is no observable conjugate field (§2), so none of the response-based instruments work. Also worth knowing before building expectations: the canonical results (Chiappori–Levitt–Groseclose *AER* 2002; Palacios-Huerta *ReStud* 2003) find professional play remarkably *close* to minimax — i.e. λ is large and the interesting structure is thin. Excellent Learn-mode content, weak research site.

---

## 2. The right criterion: does the domain have a conjugate field?

The v3 instruments are not "QRE fitters." They are response meters. Susceptibility $\chi^{\text{eq}}$ and the reciprocity defect $\mathcal{R}$ are both derivatives *with respect to a payoff perturbation* $h$. If you cannot observe a payoff perturbation, you can estimate λ but you cannot measure $\chi$, cannot measure $\mathcal{R}$, and cannot test reciprocity — which is where most of the novelty lives.

So score each domain on five things:

| | What it needs | Why |
|---|---|---|
| **A. Discrete observable actions** | choices you can histogram into $\sigma$ | everything |
| **B. Identifiable payoffs** | $U$ known analytically or estimable | HHK defence (v3 §4.1) |
| **C. Observable conjugate field $h$** | an exogenous payoff shifter you can see moving | $\chi^{\text{eq}}$, $\mathcal{R}$ — **the differentiator** |
| **D. Repeated observation** | trajectories, not cross-sections | currents, EPR, TUR |
| **E. Known potentiality** | is $\alpha$ known a priori? | calibration |

Scoring the candidates:

| Domain | A actions | B payoffs | C field | D repeat | E potential? | Engine |
|---|---|---|---|---|---|---|
| **Congestion / traffic** | route shares ✅ | **known analytically** ✅✅ | **tolls, capacity changes** ✅✅ | ✅ | **exact potential, α=0** ✅✅ | population |
| **Electricity bidding** | offer-curve steps ✅ | cost from fuel + heat rate ✅ | **fuel price shocks** ✅✅ | hourly ✅✅ | non-potential | finite-N |
| **Retail pricing** | price grid ✅ | via demand model ⚠️ | wholesale cost shocks ✅ | weekly ✅ | non-potential | finite-N |
| **Colonel Blotto** | allocations ✅ | **known by construction** ✅✅ | **experimenter-set budgets** ✅✅ | rounds ✅ | non-potential (harmonic-ish) ✅ | finite-N |
| **Auctions** | bids ✅ | values unobserved ⚠️⚠️ | reserve prices ✅ | ✅ | varies | **Bayesian (new)** |
| **Security / Stackelberg** | defender allocation ✅ | synthetic ✅ | synthetic ✅ | synthetic ✅ | varies | finite-N + leader |
| **Sports** | shot direction ✅ | approx. known ✅ | **none** ❌ | ✅ | zero-sum, λ large | finite-N |
| **Platforms** | many ⚠️ | hard ⚠️ | rare ⚠️ | ✅ | unknown | multi-stage (new) |
| **Elections** | allocations ✅ | contest function ⚠️ | spending caps ⚠️ | sparse ⚠️ | Blotto-like | finite-N |

Two domains score `✅✅` on the field criterion *and* have known payoffs: **traffic** and **Blotto**. That is not a coincidence — they are the two ends of the α axis with the payoffs handed to you. Together they are the calibration bracket for the entire programme, and neither requires estimating a demand system.

**Revised ordering** (differs from the proposal's):

- **Tier 0 — calibration bracket, build first.** Congestion/traffic (α = 0, real networks, tolls as field) and Blotto (α > 0, synthetic + experimental, budgets as field). Cheapest to build, highest instrument value, no demand estimation.
- **Tier 1 — flagship empirics.** Retail pricing (existing anchor, DreamPrice oracle) and electricity bidding (ERCOT, best real-world field via fuel shocks).
- **Tier 2 — expensive but worth it.** Auctions, once the Bayesian engine exists. Security/Stackelberg alongside it, since it shares the leader–follower machinery.
- **Tier 3 — demonstration and future.** Sports (Learn mode only), platforms, elections.

---

## 3. The engine question — the real cost of generality

The proposal's architecture sketch implies every domain is a file in `games/`. That is true for some and false for others, and the difference is expensive. There are **three distinct mathematical engines**, not one.

**Engine 1 — finite N-player strategic form (built, v3).** Players with finite action sets, mixed strategies on product simplices. All of v3 §3 applies: $S = \operatorname{blockdiag}(\lambda_i C_i)$, $B$, the resolvent $(I-SB)^{-1}$, the Hodge decomposition on the Cartesian product graph. Covers: pricing, electricity, Blotto, security, sports, elections.

**Engine 2 — population / aggregative games (new).** A continuum of small agents; the state is a distribution over strategies, not a profile of mixed strategies. This is Sandholm's setting. Consequences:
- The payoff field is $F(x)$ with Jacobian $DF(x)$; potentiality is exactness of $F$, and the susceptibility formula changes shape — the analogue of $B$ is $DF$ and the analogue of $S$ is $\lambda\,C(x)$ on the population state.
- The Hodge machinery does not transfer directly; potentiality is tested by symmetry of $DF$ (Sandholm's externality symmetry, which v3 §1.1 already flags as the known result).
- Logit equilibrium here *is* Fisk's SUE, with a known convex potential $\sum_a\int_0^{x_a}c_a(u)du + \lambda^{-1}\sum_a x_a\log x_a$.
- Covers: traffic/congestion, and any large-population routing or matching problem.

This is genuinely new core work — perhaps 30–40% of Engine 1's effort, because the solvers, the implicit diff, and the entropy machinery all carry over; only the response operators and the decomposition need re-derivation. **Worth it, because Engine 2 is where the only analytically-known potential with real data lives.**

**Engine 3 — Bayesian games with types (new, expensive).** Private values, incomplete information, interim beliefs. Needed for auctions and for common-value settings. The QRE literature has this (Goeree–Holt–Palfrey's auction work; the continuum-of-types QRE characterisation and nonparametric identification results in *GEB* 2025). Budget it as a genuine second project, not a sprint.

**Scope discipline rule, to be enforced:**

> A domain is a **plugin** iff it is exactly: a `PayoffOracle` + an action-grid constructor + a data loader + a Learn page. If it needs new response operators, new decomposition machinery, or a new equilibrium concept, it is an **engine**, and engines get their own decision.

Under that rule: pricing, electricity, Blotto, security, sports and elections are plugins on Engine 1. Traffic is a plugin on Engine 2. Auctions require Engine 3. Platforms require Engine 3 plus multi-stage machinery and should be deferred indefinitely.

---

## 4. What each Tier 0/1 domain actually gives the programme

### 4.1 Congestion / traffic — the calibration standard

**Why it matters more than the proposal realised.** Every other domain requires estimating payoffs before you can say anything, and the estimate contaminates the instrument reading. Traffic doesn't. BPR costs are given, the potential is the Beckmann integral, and logit route choice is entropy-regularised Wardrop. So you can check whether your susceptibility meter, your reciprocity meter, your current field and your EPR estimator all read the theoretically correct values **on real network data with known ground truth**. Nothing else in the programme offers that.

**Setup.** Actions = routes for each OD pair. Payoff = negative travel time. $\lambda$ = Fisk's dispersion parameter θ. Field $h$ = **link tolls**, which enter as an exactly linear additive payoff perturbation — the cleanest conjugate field available anywhere in the programme.

**Predicted readings** (things to check, not hypotheses to test): $\mathcal{R} = 0$ exactly; $J^* = 0$; EPR $= 0$; the flow response to a toll should equal $\lambda\times$ (route-flow covariance). If any of these come out non-zero, the bug is in the code, not the world — which is exactly what makes it a calibration standard.

**Data.** TNTP repo: Sioux Falls (24 zones, 76 links — debug scale, and the README explicitly warns it isn't realistic, which is fine for calibration), Anaheim (38 zones, 416 nodes, 914 links, 1,406 OD pairs), Chicago Sketch, Philadelphia, Winnipeg, Gold Coast. Format is `.tntp`; `AequilibraE` reads it and does Frank–Wolfe assignment if a baseline is wanted. Best-known equilibrium flows ship with the networks.

**Caveat.** Real-world *observed* route shares (as opposed to computed equilibria) are much harder to get than the networks are. Calibration against the known potential works with computed flows; empirical claims about actual driver dispersion need route-choice observation, which TNTP does not provide. Be clear about which you're doing.

### 4.2 Colonel Blotto — the non-potential end of the bracket

**Why.** Payoffs known by construction, budgets set by the experimenter (a perfect conjugate field), and the game is strongly non-potential with genuine mixed-strategy structure. It is the α > 0 counterpart to traffic's α = 0. It also directly enables the question the proposal poses well: does increasing non-potentiality produce measurable probability currents in resource allocation? That question is much more natural here than in retail.

**Setup.** Actions = allocation vectors on the budget simplex, discretised. Combinatorial action space — $\binom{B+k-1}{k-1}$ allocations for budget $B$ across $k$ battlefields — so this is where the matrix-free path in v3 §8.5 earns its keep. Restrict to small $(B,k)$ initially.

**Data.** Experimental literature (Chowdhury–Kovenock–Sheremeta; Arad–Rubinstein; Duffy–Matros) reports stochastic allocation and learning toward equilibrium. Raw round-level data availability is inconsistent — **check before promising empirics**; synthetic-plus-published-aggregates is the safe plan. Recent work extends Blotto to battlefield-specific variants with security-inspired instances, so the area is live.

### 4.3 Electricity bidding — the best real-world field

**Why.** Firms are explicitly, unambiguously strategic (unlike Dominick's, where the rival-firm interpretation is the weakest link in the whole programme — v3 §13). Costs are largely recoverable from fuel prices and heat rates rather than requiring a demand system. And **fuel price shocks are an excellent observable conjugate field**: a gas price move shifts one generator's marginal cost without directly shifting a coal or wind unit's, which is exactly the asymmetric perturbation the reciprocity defect needs.

**Setup.** Actions = offer-curve price steps, discretised. Payoff = (clearing price − marginal cost) × dispatched quantity, with residual demand from load minus non-strategic supply. $\lambda$ per firm/portfolio. Field = fuel cost shock.

**Data.** ERCOT 60-day DAM and SCED disclosures (unit-level offer curves, free, 60-day lag). Nord Pool aggregate bid/ask curves as a secondary check.

**Honest difficulties.** Offers are multi-part (energy, ancillary services, startup, ramp constraints) and portfolios span units, so "the action" requires a modelling choice about aggregation. Unit commitment and intertemporal constraints make the static game an approximation. The incumbent literature is supply function equilibrium (Klemperer–Meyer, Green–Newbery), which is a continuous-action framework — a discrete-grid QRE needs to justify itself against it. None of these are blockers; they are what the first three months of that plugin consist of.

### 4.4 Retail pricing — unchanged, still the flagship product application

Keeps its role from v3. The DreamPrice oracle, the Dominick's pipeline, and the entire Analyze mode are unaffected. What changes is that pricing's known weakness — the rival-firm interpretation — is now covered by having domains where strategic interaction is not in doubt.

---

## 5. Architecture changes to v3

The v3 module map survives with three modifications.

**5.1 Split games by engine, not by domain alphabetically.**

```
thermoqre/
  core/                       # engine-agnostic
    types.py  solve/  dynamics/  estimate/  viz/
  finite/                     # ENGINE 1 (v3 as written)
    games/  decompose/  response/
  population/                 # ENGINE 2 (new)
    games/                    # population game protocol, F(x)
    response/                 # susceptibility on population states
    potential/                # Beckmann/Fisk potentials, exactness test via DF
  bayesian/                   # ENGINE 3 (deferred)
  domains/                    # PLUGINS — thin by construction
    pricing/                  # oracle + grid + dominicks loader + learn page
    electricity/              # oracle + offer grid + ercot loader
    blotto/                   # oracle + allocation grid + synthetic/experimental
    congestion/               # population plugin: BPR oracle + TNTP loader
    security/
    sports/                   # learn-mode only
```

**5.2 The `PayoffOracle` protocol generalises unchanged.** This is the payoff of having chosen it in v3 §8.3. A Blotto oracle, a BPR travel-time oracle, and an electricity dispatch oracle all satisfy the same three methods. Rename `elasticity` → `response_matrix` (own/cross partial derivatives of payoff w.r.t. others' actions) so the name isn't pricing-specific; keep `elasticity` as an alias on the pricing plugin.

**5.3 Domain plugin contract** — every plugin ships exactly these five things and nothing else:

```python
# domains/<name>/__init__.py
oracle: PayoffOracle  # payoffs
grid: ActionGridBuilder  # how continuous decisions become a discrete grid
field: ConjugateFieldSpec  # what h IS in this domain, and where to find it in data
loader: DatasetLoader | None  # public data, or None for synthetic-only
learn: LearnPageSpec  # the explainer
```

The `ConjugateFieldSpec` is new relative to v3 and is the most important addition in this document. It forces every domain to declare, up front and in code, what the observable payoff perturbation is. If a domain cannot fill it in, the response instruments are unavailable for that domain and the API should say so rather than silently returning a meaningless $\mathcal{R}$.

---

## 6. What this does to the app

The Lab gains a **domain selector** above the λ and α sliders. Same instruments, different game underneath. The comparison across domains — same meters, wildly different systems — is the demo that makes the case for the whole project in about fifteen seconds.

Learn mode gains a tenth explainer: **"the same machinery everywhere"**, which runs the identical reciprocity measurement on a congestion game (reads zero), a Blotto game (reads high), and a Bertrand duopoly (reads in between). Put it last; it's the payoff for the previous nine.

Analyze stays pricing-first, because that's where the paying users are, but the upload schema generalises: entity / time / action / outcome / (optional) cost, field, competitor. A traffic planner uploading link flows and toll changes should get a sensible report from the same pipeline.

---

## 7. Revised framing

The v3 through-line was about pricing systems. Broaden it:

> Build instruments that measure how far a strategic system is from thermodynamic equilibrium, and point them at systems where the answer is known (congestion, Blotto), systems where it matters commercially (pricing, electricity bidding), and systems where nobody has looked.

The scientific spine is unchanged: potential → non-potential, MaxEnt/QRE as the mathematical backbone, payoff identification as the empirical backbone. What changes is that the α axis now has **real data at both ends** rather than synthetic games in the middle and one messy dataset at one end.

---

## 8. Build sequencing

Insert into v3 §11 as follows. Nothing here blocks anything else; take what's unblocked.

**Immediately, alongside Engine 1 core:**
- `domains/blotto` as a plugin the moment `finite/` step 7 (Hodge) lands. Payoffs are free, so it's the fastest second domain and it exercises the matrix-free path.

**After Engine 1 response operators are verified (v3 step 6):**
- Start `population/`. Beckmann/Fisk potential, population susceptibility, exactness test via symmetry of $DF$.
- `domains/congestion` + TNTP loader. This is the calibration standard; getting it early means every later instrument change gets checked against known ground truth.

**In parallel with Stream C (pricing empirics):**
- `domains/electricity` + ERCOT loader. Independent of the Dominick's work; different data engineering, same Engine 1.

**Deferred until there is a reason:**
- `bayesian/` and `domains/auction`. Revisit once Tier 0/1 has produced something.
- Platforms, elections. Not before.

**Immediate next actions, revised from v3 §14:**
1. Read arXiv:2405.07224 (unchanged — still the nearest live work).
2. Library scaffold; `reciprocity_defect()` reading 0 on a congestion game and >0 on RPS.
3. Separable Kronecker Hodge transform.
4. **New: `domains/blotto` synthetic plugin** — cheapest possible second domain, immediately gives an α > 0 anchor with known payoffs.
5. **New: skeleton of `population/` and the TNTP loader** — the α = 0 anchor with known payoffs and a real network.
6. DreamPrice decoder port (unchanged).
7. `docs/theory/01`–`10`.

With 4 and 5 in place you have the calibration bracket, both ends anchored by known payoffs, before touching a single estimated demand system. That is a much stronger position to be estimating λ from than v3 had.

---

## 9. Risks specific to this expansion

| Risk | Level | What to do |
|---|---|---|
| Engine 2 (population) is more work than the 30–40% estimate | Medium | Timebox it; if it overruns, congestion can run as a large finite-N approximation initially and lose only exactness |
| Auctions pull the project into Engine 3 prematurely | **High** | Enforce the plugin-vs-engine rule in §3. Auctions are deferred, in writing, until Tier 0/1 delivers |
| Domain sprawl dilutes the science | Medium | The plugin contract in §5.3 is the control — five files, no core changes, or it's not a plugin |
| Blotto experimental data turns out unavailable | Medium | Synthetic + published aggregates is sufficient for the calibration role; don't promise empirics until the data is in hand |
| ERCOT offer-curve aggregation choices drive the results | Medium | Report sensitivity across aggregation schemes; this is that plugin's version of the demand-specification problem |
| TNTP calibration looks like a toy to referees | Low | It is a calibration standard, not an empirical claim. Say so, and let the empirical weight sit on electricity and pricing |
| The instruments read something uninterpretable in a new domain | — | Not a risk. That's the point. Chase it |

---

## 10. Bottom line

Adopt the generalisation. Change the ordering: **traffic and Blotto first, as the calibration bracket with free payoffs and clean conjugate fields; then electricity as the best real-world strategic setting; pricing stays the flagship product.** Defer auctions until the Bayesian engine is a deliberate decision rather than an accident. And add `ConjugateFieldSpec` to the plugin contract, because a domain without an observable payoff perturbation can host λ estimation but cannot host the instruments that make this project distinctive.
