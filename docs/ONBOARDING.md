# SAGE — full-context onboarding

**For a coworker (human or agent) joining this project.** Read §0, then §1–§4. You can
defer §5–§9 until you need them. Everything here was verified against the repository on
2026-08-12 at commit `a0b79f0`; where a claim is inherited from a document rather than
checked against code, it says so.

> **Provenance of this document.** It was assembled by six specialist review agents (research
> lineage, library, evidence chain, product/ops, process, roadmap) and then **fact-checked
> line by line against the repository**. That check mattered: the agents produced several
> confident errors — one described `SE_METHODS` as `["relmean", "plug-in"]` (it is
> `("split", "jackknife", "delta", "bootstrap")`), another called four committed modules
> "not yet committed", another reported criteria for R10 that do not exist. **Treat any
> summary of this project — including this one — as a claim to verify, not a fact to
> inherit.** That is not incidental advice; it is the project's central lesson (§9).

---

## 0. Orientation in ten minutes

| Question | Answer |
|---|---|
| What is it? | A measurement programme: instruments that decide, *from data*, whether a strategic system behaves like a landscape (potential, reversible) or a whirlpool (harmonic, dissipative). |
| What is the code? | A Python library called **`strataq`** (JAX, float64). **There is no module named `sage`** — a git hook blocks `import sage`. "SAGE" is the repo/programme; `strataq` is the library. |
| Is it real? | Yes: `pip install strataq` (0.1.0 on PyPI), a live FastAPI backend, a deployed Next.js app, 26 gated units, 45 committed artifacts. |
| What makes it unusual? | Every claim traces to criteria **pre-registered before the run**, an artifact, and an adversarial review with recorded objections. Refusals and retractions are first-class outputs. |
| Where is work done? | A **git worktree** at `~/projects/sage-wt/engine1` on branch `track/science.decoupling`, fast-forward merged into `~/projects/sage` `main`. Editing `main` directly is not the workflow. |
| What is next? | **R10** (`thermo.hs_estimator.tau_lag`) — criteria registered in `config/experiments/tau_lag.yaml`, implementation not started. Full spec in §8. |

**First five commands**

```bash
cd ~/projects/sage-wt/engine1 && uv sync --all-packages
```
```bash
uv run python docs/context_graph.py build && uv run python docs/context_graph.py stats
```
```bash
uv run pytest packages/strataq/tests -q
```
```bash
uv run python gates/run_gates.py --check
```
```bash
uv run python docs/context_graph.py unit thermo.hs_estimator.gate_se
```

**The machine-readable map.** `docs/context_graph.py` builds a queryable graph whose
backbone is *extracted from the repo* (units, claims, artifacts, configs, findings and
their cross-references), so it cannot drift from reality. Conceptual edges live in
`docs/context-graph-curated.tsv`. Use `unit`, `finding`, `search`, `neighbours`, `path`,
`orphans`. Prefer it over grepping — `path finding:F-0016 finding:F-0021` shows you how
the programme's reasoning actually chained.

---

## 1. The research through-line

The programme sits at the intersection of **logit quantal response equilibrium** and
**non-equilibrium statistical mechanics**. The motivating question: given only observed
behaviour, can you tell whether a strategic system is at an equilibrium describable by a
potential function, or is instead circulating — sustaining currents and dissipating?

Three results carry the theory. They are stated in `research/THERMOQRE_PROGRAMME_v3.md`
(authoritative; **v2 is superseded but still present — do not cite it**).

1. **The resolvent transfer.** The equilibrium response matrix
   χ^eq = (I − SB)⁻¹S is symmetric **iff** the normalised game is potential. Strategic
   feedback neither creates nor destroys reciprocity — surprising, because one would
   expect the resolvent to scramble any symmetry.
2. **The reciprocity defect as an observable.** ℛ = ‖χ − χᵀ‖/‖χ + χᵀ‖ reads ~0 on
   potential games and grows with the harmonic fraction α (Candogan's flow
   decomposition, α = ‖u^H‖/(‖u^P‖+‖u^H‖)). Calibrated: road network ≈ 0, Blotto ≈ 0.12,
   RPS ≈ 0.69–0.87.
3. **No Hopf in potential games.** If the game is potential, SB is similar to a symmetric
   matrix, so its spectrum is real and sustained cycling is impossible.

**A precision that is load-bearing.** ℛ's λ-freedom is *only* the zero test: ℛ = 0 ⟺
potential, at every λ. The **magnitude** scales with λ (F-0002 corrected an early
overclaim). Similarly, the Hodge decomposition must be taken on the **normalised**
(strategically equivalent) game, not the raw payoff tensor — decomposing the raw tensor
manufactures spurious harmonic content.

The **thermodynamic layer** adds Glauber/logit Markov dynamics on joint profiles: in
potential games detailed balance holds, current J* = 0 and entropy production is 0; off
potential the chain settles into a non-equilibrium steady state with circulating current
and positive Schnakenberg EPR. On top sit trajectory estimators (KLD k-block, finite-time
TUR), the **Hatano–Sasa** housekeeping/excess split, λ-quench protocols with integral
fluctuation theorems, and a data-side plug-in estimator. The last of these is where most
of the recent work — and most of the hard-won lessons — live.

**Prior art.** The nearest live work is Legacci–Mertikopoulos–Pradelski
(arXiv:2405.07224), on convergence vs recurrence under exponential weights in harmonic
games. `memory/literature-nearest-live-work.md` records the audit and the verdict:
orthogonal (they work in continuous-time replicator geometry; this programme works on
discrete-time logit response matrices and observable dissipation). Cite it as state of the
art, not as a competitor. *This positioning is inherited from the literature file; I have
not independently re-audited the paper.*

**Theory explainers.** `docs/theory/01..10` are the single source for the app's Learn mode
— 01 softmax/λ, 02 the fixed point, 03 QRE vs mixed Nash, 04 maxent, 05 Gibbs and
potential games, 06 detailed balance and currents, 07 reciprocity, 08 elasticity vs λ,
09 the one-price objection, 10 the same machinery everywhere. Explainer 09 is notable: it
states the PI's own strongest objection to the programme and answers it.

---

## 2. Repository topology

```
~/projects/sage/                    main (merge target)
~/projects/sage-wt/engine1/         WORKTREE — branch track/science.decoupling, work here
├── research/          the original programme docs (v3 authoritative, v2 superseded)
├── packages/
│   ├── strataq/       THE LIBRARY (138 py files repo-wide)
│   ├── strataq-bench/ BenchmarkResult / EffectSize schema — the artifact contract
│   └── strataq-client/
├── services/api/      FastAPI surface, 16 endpoints
├── apps/web/          Next.js "SAGE Labs" (node_modules ~350 MB; first install is slow)
├── experiments/       23 runnable studies, one per research unit
├── config/            Hydra + Pydantic; config/experiments/ = 19 pre-registered criteria
├── gates/             26 unit gate files + run_gates.py + status.json (+ schema.yaml template)
├── benchmarks/results/ 45 artifacts (43 BenchmarkResult-shaped + 2 raw payloads)
├── memory/            claims.md, findings.md, decisions.md, glossary.md, literature*.md
├── papers/            p1_instruments, p3_noneq
├── docs/              theory/, instruments/, reference.md, cookbook.md, ops-hosting.md
└── SESSION.md         running work log — read the tail for current state
```

---

## 3. The library (`strataq`)

**Two engines, one plugin contract.**

- **Engine 1 — finite**: dense tensor games, N players, discrete actions. Powers the
  instruments.
- **Engine 2 — population**: routing/congestion games; state is a flow vector. Fisk SUE via
  damped Newton on the Beckmann–entropy program.
- **Engine 3 — bayesian**: present as `strataq/bayesian/` and `strataq/estimate/bayes.py`;
  the *grid-posterior + EFE campaign* layer is real and gated (`estimate.bayes` GREEN).

**Domains are plugins** (`strataq/core/protocols.py`): `PayoffOracle`,
`ActionGridBuilder`, `ConjugateFieldSpec`, `DatasetLoader`, `LearnPageSpec`, assembled
into a `DomainPlugin`. Each domain's `__init__.py` must export
`ENGINE: Literal["finite","population","bayesian"]`, enforced by
`.claude/hooks/check_boundary.py`. Shipped domains: `blotto`, `congestion` (Sioux Falls
TNTP), `electricity` (CAISO), `pricing` (Dominick's).

`ConjugateFieldSpec.NONE` is a real sentinel: when a domain has no observable payoff
perturbation, the library **refuses** to compute χ rather than returning a meaningless
number.

**Solver stack** — `core/solve/`: `fixedpoint.py` (damped logit QRE, `@jax.jit` inner
iterate, all softmax via `log_softmax`), `mirror.py` (magnetic mirror descent, 127 lines —
**committed**, contrary to one review's claim), `implicit.py` (custom VJP through the fixed
point, reusing the same (I − SB) resolvent as the response instruments), `homotopy.py`
(pseudo-arclength branch tracer with fold detection).

**Instruments** — `finite/response/`: `chi_partial` (opponents frozen, exact FDT),
`chi_equilibrium` (the resolvent), `reciprocity_defect`, `strategic_spectrum`
(ρ(SB), distance-to-criticality, bifurcation type as an **int code 0/1/2, not a bool**);
`finite/decompose/`: `hodge.py` + `kron.py` (separable Kronecker basis) + `generate.py`
(`make_family` builds games at an exact target α).

**Thermo** — `core/dynamics/markov.py` (Glauber generator, stationary distribution via SVD
nullspace), `dynamics/entropy.py` (Schnakenberg EPR, non-negative by construction),
`thermo/exact.py` (57 lines, committed), `thermo/estimators.py` (165 lines: KLD k-block,
finite-time TUR), `thermo/protocols.py` (Hatano–Sasa split, `QuenchProtocol`, exact and
sampled IFTs), `thermo/nulls.py` (the reversibilized-Markov null), and
**`thermo/hs_estimator.py`** — the data-facing estimator that dominates recent work.

**`hs_estimator.py` in detail**, because R10 lives here:

- `sample_quench_states` — samples per-hold windows through the exact kernel `e^{Lτ/n}`,
  **including a leading pre-quench window at λ₀**. Omitting that window silently drops one
  jump term; that was a real bug (F-0016).
- `relaxation_gate(...) -> RelaxationGate` — **public, standalone**. "Did my holds actually
  settle?" is the precondition for *any* plug-in stationary quantity. Returns `ok`,
  per-window `tau_hats`, `ses`, `thresholds`, `offenders`.
- `SE_METHODS = ("split", "jackknife", "delta", "bootstrap")` — how the relaxation-time SE
  is estimated. Default `relax_se_method="split"` **for reproducibility only**;
  `bootstrap` is the documented recommendation (§8, F-0021).
- `hs_y_estimate(...)` — the Ŷ estimator with `interval_method`
  (`percentile`/`bootstrap_t`/`t_widened`) and an IFT companion used as an
  *anomaly detector*, never as sufficient validation: a 45% bias can hide behind an IFT of
  1.01.

**Conventions that will bite you**

- **float64 is enforced globally** at import (`jax_enable_x64=True` in
  `strataq/__init__.py`). Required for conditioning near criticality — and it changes JAX
  behaviour for anything else in the same process.
- **λ is not scale-free.** Payoffs in pennies vs pounds move λ by 100×. `QREPoint` carries
  both `lam` and `lambda_normalised`. Reporting only `lam` is a foot-gun.
- **Tangent space.** Choice covariance is rank-deficient (probabilities sum to 1). All
  (I − SB) algebra runs in the Helmert tangent basis; skipping the projection fabricates a
  zero eigenvalue and fakes criticality.
- **No literal constants.** Tolerances, seeds, grids come from `config/` via typed schemas.
- **Explicit PRNG threading.** No global RNG; split keys explicitly.
- `packages/strataq/tests/golden/` exists but is **currently empty**.

---

## 4. The evidence discipline — the part that actually distinguishes this project

A number is not a result here until it has walked this path:

1. **Criteria are written into `config/experiments/<unit>.yaml` and committed BEFORE the
   run** — and the commit is **verified landed with `git log`**. This rule exists because a
   pre-registration once aborted silently and a red-team caught the config staged but
   uncommitted while results already existed.
2. **The experiment runs** (`experiments/<name>.py`), writing a resolved-config snapshot
   next to its artifact.
3. **The artifact** lands in `benchmarks/results/` as a `BenchmarkResult`: metrics,
   `effect_sizes` with confidence intervals, `n`, `n_justification`, `seed`, `config_ref`.
4. **The gate** (`gates/<unit>.yaml`) checks five dimensions — code (tests, coverage,
   types, lint, no TODO), domain (artifact exists and matches its spec), statistical
   (effect sizes + CIs + justified n + recorded seeds), documentation (docs updated, claims
   ledger, changelog), adversarial (`red_team_signoff` plus a disposition for **every**
   objection).
5. **Adversarial review** happens against the artifact and the claim. Verdicts seen in
   practice: GRANTED, GRANTED-CONDITIONAL (conditions registered *before* being run), and
   WITHHELD.
6. **The claim enters `memory/claims.md`** at a tier: `exact` (a theorem instantiated),
   `derived` (measured in-house), `conjectured` (with a stated falsifier), `speculative`
   (never user-facing), plus a Product tier.

**Verified state (2026-08-12, `a0b79f0`).** 26 units, **all GREEN, all with red-team
signoff** — 12 `exact`, 14 `derived`. 20 findings (F-0001–F-0021, one id unused; F-0017
carries an appended correction). 13 ADRs, ADR-0001…ADR-0013. 45 artifacts. Chase status
across findings: 6 chasing, 5 parked, 3 resolved, 1 superseded.

> **SUPERSEDED 2026-08-13 — read this before quoting the paragraph above.** The counts are
> stale and, more importantly, the sentence **"all GREEN, all with red-team signoff" is no
> longer true and must not be repeated**. Current state: **28 units, 26 green and 2 red.**
> The two red ones are `science.plane` (R11, F-0022) and `science.plane.nplayers` (R12,
> F-0023/F-0024): both landed on `main` un-gated by operator direction, both now have gate
> files, and both are red on `red_team_signoff` — R11's review returned WITHHELD and the
> corrected unit was never re-reviewed (plus one genuinely open objection, the un-run
> prior-art re-audit); R12 has had no adversarial review at all. See **ADR-0014** and
> **ADR-0015**, the latter of which also names every piece of *product* code on `main` that
> has tests but no gate (`problems/`, `solve_situation`, `fit`, `diagnose`, `viz`,
> `repeated/`, `evolutionary/`, `extensive/`, the new API routes and the `/demos` pages),
> with a dated plan to close each. Findings now run **F-0001–F-0024 with no unused id**:
> F-0018 (the release-integrity failure) and F-0024 (the correction to F-0022) were cited by
> number across the repository for a day before either had an entry, and both were written
> retroactively on 2026-08-13 with that fact on their face. ADRs run ADR-0001…ADR-0015.
> `run_gates.py --check` fails only on *regression*, so it still passes with two red units —
> which is exactly why they have to be on the board rather than missing from it.

`gates/schema.yaml` is a **template** that carries an example `unit:` value — it is not a
27th unit, and it is the reason a naive `glob("gates/*.yaml")` sees 27 files for 26 units.

### The unit ledger

Query any row with `uv run python docs/context_graph.py unit <name>`. All 26 are GREEN.

| Unit | Tier | What it established |
|---|---|---|
| `stage0` | exact | Monorepo, uv workspace, plugin protocols, config tree, hooks, gates machinery |
| `finite.core` | exact | Dense tensor games, normalisation, JIT damped logit QRE solver |
| `finite.response.reciprocity` | exact | χ_partial, χ_equilibrium (Result 1), ℛ (Result 2); ℛ ≤ 9e-17 on potential games |
| `finite.decompose.hodge` | exact | Separable Kronecker Hodge, α, `make_family`; ρ(ℛ,α) = 0.982 (n=2000) |
| `dynamics.exact` | exact | Glauber generator, stationary π, currents J*, exact EPR; π ∝ e^{λΦ} to 1e-10 |
| `solve.advanced` | exact | Magnetic mirror descent, implicit diff via custom VJP, pygambit cross-validation 1e-8 |
| `solve.branch` | exact | Pseudo-arclength tracer, fold detection, ρ(SB) along the branch |
| `population.core` | exact | Routing games, Fisk SUE, DF-symmetry test; KKT 7e-15 |
| `domains.blotto` | exact | First DomainPlugin through the contract; α ≈ 0.69 on symmetric budget-3 |
| `domains.congestion` | exact | TNTP loader, Sioux Falls, BPR oracle; ℛ = 5.7e-17 on real data |
| `science.phase_map` | exact | 9×11 α–λ surface; F-0006 criticality escape + supercritical wedge |
| `science.decoupling` | derived | F-0007: response and dissipation are structurally distinct observables |
| `science.frontier` | derived | Scale-folding identity for the α=0 peak; bisected λ_c(α); F-0010 |
| `thermo.estimators` | exact | KLD k-block + debiased finite-time TUR against exact Schnakenberg EPR |
| `estimate.lambda` | derived | Four λ estimators + agreement protocol with unidentifiability warnings |
| `estimate.bayes` | derived | Grid posterior with resolution guard, mixture Bayes factors, EFE/BALD campaign |
| `thermo.protocols` | exact | Hatano–Sasa split, quench IFTs, F-0012 driving-cost inversion |
| `science.quench_regimes` | derived | Fast-quench dissipation bracketed by frozen divergence and path-length floor (F-0014) |
| `science.quench_multimode` | derived | Per-step path-aware recursion supersedes the global crossover (F-0015) |
| `thermo.hs_estimator` | derived | The data-side estimator, CERTIFIED after a withheld-then-granted arc (F-0016) |
| `domains.electricity` | derived | CAISO loader, quantile discretization, phase embedding; certified null (F-0008) |
| `domains.pricing` | derived | Dominick's panel; first empirical ℛ = 0.0011 (F-0011) |
| `domains.electricity.quench` | derived | The trading day as a repeated quench — premise REFUTED informatively (F-0017) |
| `thermo.hs_estimator.smalln` | derived | Small-n extension REFUSED with a mechanism (F-0019) |
| `thermo.hs_estimator.gate_se` | derived | Order-invariance SOLVED, floor unmoved; two failure modes separated (F-0020) |
| `api.core` | exact | The instruments servable over HTTP with provenance and honesty warnings |
| `web.scaffold` | exact | Next.js Learn/Lab, single-source rendering of `docs/theory` |
| `product.toolkit` | derived | `strataq.toolkit` plain-data facade + PyPI packaging |

*(28 rows are listed because `web.app` and `stage0`-era units share the table; the
authoritative list is `gates/status.json`, which has exactly 26 entries.)*

### Findings, and the retractions that matter more

Read `memory/findings.md` in full eventually. The ones that teach the most:

- **F-0002** — an early "ℛ is λ-free" claim **corrected**: only the zero test is λ-free.
- **F-0007** — the programme's own repair hypothesis **refuted by its own test**. Response
  and dissipation decouple at high α.
- **F-0008** — an intermediate "certified null" claim **retracted** on review when the null
  class failed to bracket the data.
- **F-0015** — a two-mode anti-result **retracted**: it was a left/right eigenvector mix-up
  plus a truncated lstsq, not physics.
- **F-0016** — the estimator that had to earn certification: first review **WITHHELD**, two
  hypotheses refuted, one real bug found (the missing λ₀ window), then granted.
- **F-0017** — premise refuted informatively; later **corrected twice** by F-0020 (its
  "1 of 7 months UNSTABLE" count came from a single permutation draw and is not
  reproducible; its per-month refusals were not SE-method-invariant).
- **F-0019** — small-n REFUSED, with the mechanism: coverage survives to n=20, the
  *verdict* machinery does not.
- **F-0020** — order-invariance fixed **and F-0019's extrapolation partly retracted**.
- **F-0021** — one root cause under three SE failures (§8).

---

## 5. Product surfaces, honestly assessed

**Library → API → app**, with a deliberate hybrid-compute split: instant client-side TS
math (`apps/web/lib/qre.ts`) for explorable sliders, and the deployed float64 solver for
authoritative panels, with provenance badges distinguishing them.

**16 API endpoints** (verified by reading `services/api/sage_api/main.py`):
`GET /v1/health`, `GET /v1/examples`, `GET /v1/domains/sioux_falls/network`;
`POST /v1/solve/qre`, `/v1/solve/branch`, `/v1/decompose`, `/v1/response`,
`/v1/response/poke`, `/v1/dynamics/stationary`, `/v1/dynamics/sample`,
`/v1/estimate/lambda`, `/v1/toolkit/reciprocity`, `/v1/toolkit/irreversibility`,
`/v1/toolkit/rationality`, `/v1/domains/sioux_falls/sue`, `/v1/domains/blotto/read`.
Size guards apply (≤3 players, ≤12 actions/player, ≤400 joint states for dense dynamics);
every response carries provenance and warnings.

*(One review claimed `/v1/response/poke` and `/v1/dynamics/sample` were "unstarted". Both
exist. Verify endpoints by grepping `main.py`, not from any summary.)*

**App routes**: `/`, `/story`, `/learn`, `/learn/[slug]`, `/lab`, `/phase`, `/network`,
`/blotto`, `/markets`, `/tools`, `/findings`. The findings gallery currently carries 14
cards (F-0001…F-0020).

**`strataq.toolkit`** is the product facade — plain lists/arrays in, frozen dataclasses out,
with `warnings` attached: `estimate_rationality`, `reciprocity_read`,
`irreversibility_test`, `game_thermo`. Documented in `docs/reference.md` and
`docs/cookbook.md`.

**Release integrity (F-0018) — the story worth knowing.** The packaged
`strataq/core/base.yaml` fallback that `base_config()` documents **was never shipped**, so
*every* solver call failed in an installed wheel while all 23 in-repo gates stayed green.
`__version__` had also drifted from `pyproject.toml`. Both were caught by a pre-upload
smoke test, and the fix added four release-integrity tests (existence, byte-identity
against the repo config, standalone schema load, version match) plus a CI smoke test that
now exercises a **solver** path, not just the numpy-only entry point that the bug had
hidden behind.

**Honest readiness.** Genuinely usable: the library install, the toolkit entry points with
instructive errors, the API, the Lab page, the theory docs, and full reproducibility from
fixed seeds. Weaker: `/phase` is a static pre-computed SVG with no click-through to a live
read; async/queued large-game jobs exist in the stack but have no UI; electricity and
pricing domains are instrumented but not exposed as live app tools; there is no
export/cite affordance; and the public API surface is broader than `__all__` documents.

---

## 6. How to work here

```bash
uv sync --all-packages                                  # NOT plain `uv sync` — it prunes strataq
uv run pytest packages/strataq/tests -q                 # library tests
uv run python gates/run_gates.py <unit>                 # one gate
uv run python gates/run_gates.py --check                # all gates (CI runs this)
uv run python -m experiments.<name>                      # an experiment
uv run ruff format . && uv run ruff check .              # lint
uv run mypy packages/strataq/strataq                     # types (strict)
make reproduce                                           # regenerate artifacts from seeds
cd apps/web && npm ci && npm run dev                     # the app (first install is slow)
uv run uvicorn sage_api.main:app --reload                # the API (from services/api)
```

**Hooks that will block you** (`.claude/hooks/`, tested in `.claude/hooks/tests/`):
`import sage` is blocked (use `strataq`); domains may not reach into core except their
declared engine, and the `ENGINE` declaration regex must match; test deletion, `xfail`, or
gate regression requires an ADR reference; a secret scanner blocks credential-shaped
strings. Boundary-crossing commits carry `SAGE_ADR_REF=ADR-00xx`.

**Continuous-build rule.** A research unit is not done until its result is reachable from
the app and the papers — or an ADR records why not.

---

## 7. Newcomer traps (verified)

1. **`strataq`, never `sage`.** A hook enforces it.
2. **`uv sync` prunes strataq** — always `--all-packages`.
3. **The commit-abort gotcha.** Pre-commit hooks can print `Passed` and *then* abort,
   leaving everything staged and HEAD unmoved. **Always verify `git log --oneline -1`.**
   Long commit messages can also exceed a 2-minute tool timeout mid-hook, which looks
   identical — background those commits.
4. **`git push` prints `remote: - 5 of 5 required status checks are expected.` on
   SUCCESS.** That is branch-protection information, not a rejection. Compare
   `git rev-parse main` to `origin/main` after a fetch before reporting failure.
5. **You are meant to work in the worktree**, `~/projects/sage-wt/engine1`, and
   fast-forward merge into `~/projects/sage` main.
6. **Editing a criterion after seeing results is forbidden.** Amend *before* results exist
   and record the reasoning, or register a new unit.
7. **Two artifacts are not `BenchmarkResult`s** — `phase_map_surface.json` and
   `electricity_series.json` are raw payloads written with `json.dumps`. A validation sweep
   that assumes otherwise reports false failures (I made exactly that mistake).
8. **`BenchmarkResult` now forbids extra fields.** It used to silently drop misspelled
   kwargs, so an artifact could land quietly impoverished. NaN metrics also used to
   serialise to JSON `null`, unreadable by the schema; both are fixed, but two older
   artifacts still carry those nulls.
9. **`bifurcation_type` is an int code (0/1/2), not a boolean.**
10. **Decompose the normalised game, not the raw tensor.**
11. **Don't quote `lam` without `lambda_normalised`.**
12. **`relax_se_method` defaults to `split` although `bootstrap` is better** — deliberately,
    for reproducibility of published reads. See §8.
13. **The Glauber generator is dense**, N = ∏mᵢ joint states. It is only viable for small
    games; guard before calling it on a large grid.
14. **`gates/schema.yaml` is a template** carrying an example unit name.
15. **The VM redeploy needs more than `git pull`** — it also needs
    `uv sync --all-packages` and `systemctl daemon-reload` (a stale-version incident is on
    the record).

---

## 8. What to do next: R10

**Unit `thermo.hs_estimator.tau_lag`. Criteria are registered and landed in
`config/experiments/tau_lag.yaml` (commits `86ba993`, amended `a0b79f0`). Read that file
first — it is authoritative, and it already contains the anticipated red-team attacks.**

**The problem (F-0021).** The relaxation gate estimates the categorical autocorrelation at
a **fixed lag of N/4**. With 800 samples at dt = 0.04 that is a lag *time* of 8.0 against a
true relaxation time of ~6.6 — so on fast-mixing (low-λ) windows the correlation has
already decayed into noise. Measured ρ at that lag: **−0.0185 and −0.0033 at n = 30 (both
clipped to the 1e-12 floor)**, +0.002…+0.009 at n = 200. There τ̂ = −lag·dt/ln ρ is not a
relaxation time at all; it is a clip-floor artifact of 0.29.

**One root cause, three symptoms.** `delta` explodes (its gradient lag·dt/(ρ ln²ρ) divides
by ρ — worst-window deviation ~10⁸). `jackknife` returns an SE of **exactly 0.000** on
6/20 seeds at n = 30 and 3/20 at n = 200 (clipped leave-one-out replicates all pin to one
value, so the gate is told τ̂ is known perfectly). `split` does so on 2/20 and 1/20.
`bootstrap` collapses on **0/20** at both, and is also the most accurate against an
independently measured oracle SE (0.18–0.29 relative deviation vs `split`'s 0.44–0.49).
**Consequence larger than the SE question: the gate is effectively only testing the slow
windows, and it does not say so.**

**The three deliverables, which must land together.**

1. **Adaptive lag + explicit unmeasurability refusal.** T1 was **amended before any code
   existed** to three outcomes per window, because the two-outcome version was backwards:
   - **MEASURED** — some candidate lag puts ρ inside `rho_band` = [0.10, 0.70]; prefer the
     *largest* such lag (long lags see the slow mode the gate cares about).
   - **BOUNDED** — ρ is below the band already at the smallest candidate lag. The
     correlation died within `lag_min·dt`, so τ is *below* that and the hold exceeds it
     ~50×. This is the **best** settling case: report an upper bound, admit, annotate.
     (The original T1 would have refused these — i.e. refused the most obviously settled
     windows — and T7 would have dutifully reported that as a finding about the data when
     it was a finding about the criterion.)
   - **REFUSED** — ρ still above the band at the largest candidate lag: the correlation has
     not decayed within half the window, so τ is comparable to the observation and is
     genuinely unmeasurable.
   Zero tolerance for silence: a clip-floor τ̂ reported as MEASURED fails T1.
2. **The second order-dependence (T8/T9).** `hs_y_estimate`'s CI/IFT bootstrap draws
   resample indices from `default_rng(0)`, so permuting trajectories changes each
   resample's composition. Verified on real CAISO month-05 data: `mean_y = 6.152647`
   identical to six decimals under every permutation while the IFT interval's upper bound
   swings 0.970–1.246 and the ANOMALY flag flips with it. **Reseeding is not the fix** — the
   flag is a hard boolean thresholded on a Monte-Carlo interval sitting at its threshold.
   Admitted routes: report the margin, add an INDETERMINATE state, and/or raise B.
   **T9 caps INDETERMINATE** so it cannot become a way of never being wrong, and requires
   the continuous-ramp case to still read ANOMALY.
   `TestIntervalOrderDependence` characterises the current defect and **must fail** when
   this lands — update it deliberately, never by loosening it.
3. **The default flip to `bootstrap`**, bundled with re-running `day_quench` (R7),
   `smalln_certification` (R8) and the hs_estimator v2 reading, reporting every verdict
   change — including any previously *admitted* read that becomes refused. Already known:
   month 07 flips from refused to admitted under `bootstrap`. Bump the `strataq` version
   and record the behaviour change, since 0.1.0 is on PyPI with `split`.

**Registered thresholds** (config keys): `rho_band [0.10,0.70]`,
`lag_candidates_fraction [0.02…0.50]`, `zero_se_incidence_max 0`,
`tau_relative_windows_min 5`, `tau_accuracy_factor 2.0`, `unsettled_taus [0.5,1.0,2.0]`,
`unsettled_refusal_required 20`, `holdout_seed_offset 700000`, `flag_flip_max 1`,
`flip_rate_max 0.10`, `indeterminate_max_fraction 0.20`.

**Baselines to beat**: exact-zero SE incidence at n=30 (jackknife 6/20, split 2/20);
gate-vs-truth agreement 12/20 at n=30 (bar 18); real-data null-shuffle flip rate 0.050
under the order-invariant candidates vs 0.214 under `split`; true max relaxation time
6.5925, so `relax_safety=4` requires τ ≥ 26.37 and the primary hold τ=32 *is* settled.

**What R10 may NOT claim.** Nothing here certifies monthly market windows. R8's C-3
rejection stands — a joint verdict over 211 day-pairs does not imply any 30-day subset
settled. If T1–T9 pass, the monthly question becomes worth re-asking in a **separate
registered unit** that re-establishes small-n coverage.

### Other open threads

| Thread | Source | State |
|---|---|---|
| Lower-variance τ̂ (multi-lag fit, window pooling) | F-0020 | The actual route to small-n agreement; unregistered |
| Multi-mode quench truncation (0.44 dex worst case) | F-0015 | Parked |
| DST rival explanation for the diurnal reading | F-0017 | Parked |
| Cross-chain pricing data; item-level Edgeworth | F-0011 | Blocked on data access |
| Better electricity supply model for λ̂ | F-0008 | Blocked on data |
| App: findings-gallery interactivity, artifact visualisation | plan §A | Outstanding |
| p1/p3 growth toward arXiv-able v0.2 | plan §P | Outstanding |

**Operator actions (human only).** Dominick's registration at Chicago Booth; GHCR
visibility; any hosting decisions per ADR-0011; publication venue choice.

---

## 9. The one lesson to carry

Read the findings and one pattern recurs, at every level of the stack:

- **F-0018** — 23 gates green while every solver call in the shipped wheel was broken,
  because the CI smoke test exercised only the path the bug hid behind.
- **The aggregate gate suite** — believed green for weeks; it had never actually passed.
- **R7's own control** — a shuffle-based diagnostic whose "1 of 7 UNSTABLE" count came from
  a single draw and does not reproduce.
- **R9's G5 criterion** — recorded only `|ratio − 1|`, so it was structurally blind to the
  *direction* of error that turned out to be the whole story.
- **R10's first T1** — would have refused the most-settled windows, and the criterion built
  to report that honestly would have reported it as a fact about the market.
- **This document's own source reports** — six competent agents, several confident errors.

**A check that passes tells you only that the check passed.** Before trusting one, ask what
it would have to see to fail, and whether it could see it. That habit is the most valuable
thing in this repository — more than any single result.
