# Decisions — ADRs

Format per entry: Context · Options · Decision · Consequences · Date. Includes every divergence from the research docs. Referenced by hooks via `SAGE_ADR_REF=<id>`.

---

## ADR-0001 — Repo/package naming: repo `sage`, package `strataq`

- **Context**: SageMath owns the top-level `sage` import; publishing or importing `sage` is unusable.
- **Options**: (a) rename everything; (b) repo `sage` + package `strataq`.
- **Decision**: (b), per the master build prompt. `import strataq` always; never publish `sage` to PyPI. Enforced by hook + CI + test (`test_import_guard.py`).
- **Consequences**: A permanent naming split between repo and package; every doc states it once.
- **Date**: 2026-08-08

## ADR-0002 — Fresh monorepo at `~/projects/sage`, not in `~/projects/QRE`

- **Context**: The build prompt assumes "the sage repo"; the session started in `~/projects/QRE`, which held only the research docs. GitHub `SharathSPhD/sage` existed, empty.
- **Options**: (a) git-init QRE in place; (b) fresh `~/projects/sage` wired to the existing remote.
- **Decision**: (b), confirmed by the PI (2026-08-08). QRE stays untouched as the source archive; docs copied into `research/` (with `research1.md` as a plain-text conversion of the RTF).
- **Consequences**: Future sessions should be started from `~/projects/sage`.
- **Date**: 2026-08-08

## ADR-0003 — Python version policy

- **Context**: `requires-python >= 3.11`; only 3.12 installed locally.
- **Decision**: Local dev pinned to uv-managed 3.12 (PI-confirmed); CI matrix 3.11/3.12/3.13 is authoritative for compatibility. mypy runs under the active interpreter (no `python_version` pin — 3.11 pinning breaks numpy's 3.12-syntax stubs; the CI matrix covers 3.11 semantics).
- **Date**: 2026-08-08

## ADR-0004 — Engine 3 (`bayesian/`) is deferred, in writing

- **Context**: Auctions require type spaces (DOMAINS v1 §3): the finite-strategic-form machinery (S, B, resolvent, Hodge) does not apply as written. High risk of premature core sprawl.
- **Decision**: `bayesian/` is not started until Tier 0/1 domains have produced results and a superseding ADR exists. The directory exists as a marker only.
- **Consequences**: Auctions and platforms are out of scope for Stages 0–4. Anyone proposing a domain that needs types is proposing an engine.
- **Date**: 2026-08-08

## ADR-0005 — Package layout: `packages/strataq/strataq/…`

- **Context**: The master prompt's schematic layout shows modules directly under `packages/strataq/`. Python packaging convention puts the import package inside the distribution directory.
- **Decision**: Conventional layout (`packages/strataq/strataq/core/…`) with hatchling `packages = ["strataq"]`. Divergence is cosmetic; module map (core/finite/population/bayesian/thermo/domains/data) is exactly as specified.
- **Date**: 2026-08-08

## ADR-0006 — Dominick's licence handling

- **Context**: `qbz506/dreamprice-dominicks-cso` and all derived artefacts are CC-BY-NC-4.0; library is Apache-2.0.
- **Decision**: Licence rides with the data, enforced at the loader and in every dataset card. No Dominick's-derived bytes in the Apache-licensed package or its tests; loaders fetch at runtime. `research/` docs record the intent; `domains/CLAUDE.md` carries the rule.
- **Date**: 2026-08-08

## ADR-0007 — N3 tier upgrade (conjectured → derived)

- **Context**: Prior-art sweep (2026-08-08, theory-verifier, `literature-nearest-live-work.md`): the similarity argument is sound and unpublished for logit; Hommes–Ochea 2012 supports the contrapositive.
- **Decision**: N3 recorded as `derived` in `claims.md`, with numerical verification explicitly outstanding (α-sweep). A cycle at α = 0 reverts the tier hard and is a headline finding.
- **Date**: 2026-08-08

## ADR-0008 — Boundary rule refinement: plugins may import their own engine

- **Context**: The original hook blocked every `strataq.finite/population` import from `domains/`. The congestion plugin cannot exist without the population engine's `RoutingNetwork` — a plugin *runs on* an engine (DOMAINS v1 §5.3 says plugins sit on engines; it forbids plugins needing NEW core machinery, not using existing engine objects).
- **Decision**: `check_boundary.py` reads the domain's declared `ENGINE` and allows imports from that engine's package only. Cross-engine imports and cross-domain imports remain blocked; commits touching both domains and core still require an ADR reference.
- **Consequences**: The contract's real invariant — "a domain that needs core *changes* is an engine" — is unchanged and still enforced by the mixed-commit check.
- **Date**: 2026-08-08

## ADR-0009 — ADR escape hatch readable from commit messages in CI

- **Context**: CI re-runs the boundary check on pushed commits, where the local `SAGE_ADR_REF` env var no longer exists; the calibration-bracket merge (legitimately referencing ADR-0008) failed CI's guard.
- **Decision**: `check_boundary.py` falls back to scanning the HEAD commit message for `ADR-\d{4}` tokens that exist in `memory/decisions.md`. Same strictness (the ADR must exist), better auditability (the override is recorded in history, not a transient env var).
- **Date**: 2026-08-08

## ADR-0010 — Backend hosting: Oracle Always Free VM replaces Render

- **Context**: PI decision recorded in SAGE_HOSTING.md (now docs/ops-hosting.md): Oracle A1.Flex 2 OCPU / 12 GB ARM at EUR 0/month; research compute stays on the DGX Spark; Supabase/Vercel/HF unchanged.
- **Decision**: services/api deploys to the Oracle VM via Docker Compose + Caddy (deploy/); CI builds a linux/arm64 image to GHCR. render.yaml retained as the documented fallback path only (Netcup is the paid fallback). Provisioning: VCN sage-vcn + subnet created 2026-08-10 (us-ashburn-1); instance launch is capacity-gated with a scripted retry (the documented Always Free failure mode).
- **Consequences**: worker concurrency 1 on 2 ARM cores; game-size limits enforced in API validation before queueing; arm64 images mandatory.
- **Date**: 2026-08-10

## ADR-0011 — Interim backend on Always Free x86 micro; A1 retry continues; PAYG declined

- **Context**: A1 capacity exhausted in all Ashburn ADs; the PAYG upgrade path (community-reported to improve A1 queue position — not guaranteed) was declined because Oracle's card verification attempts a £79 hold. Free Tier cannot subscribe to other regions (home region fixed at signup), so "another region" is not an option without PAYG.
- **Decision**: Provisioned VM.Standard.E2.1.Micro (Always Free x86, 1 GB + 4 GB swap) — capacity available immediately, different pool from A1. API deployed bare-metal (uv + systemd + Caddy; no Docker at 1 GB) at http://150.136.84.2 with reduced size limits (10 actions/player, 200 profile states). The scripted A1 retry keeps running; on an A1 landing, migrate via deploy/ compose (portable by design). If A1 hasn't landed in ~48h, Netcup/GCP per the PI.
- **Consequences**: interim box is slow (1/8 OCPU burstable) and RAM-tight — fine for Lab-scale sync calls; branch tracing and dense dynamics stay conservative. Ops detail recorded: Oracle's default iptables REJECT sits at INPUT position 5 — port rules must be inserted ABOVE it (the hosting doc's gotcha, met in practice).
- **Date**: 2026-08-10

## ADR-0012 — R6 decision: Bayesian inference engine next; deeper empirics wait for data

- **Context**: Plan v2's R6 gate said "decide after R2/R4 readings exist". They now do, and both hit data walls, not method walls: R2 (electricity) rejected its stylised auction model outright for λ (F-0008/F-0009 — the dissipation reading stands, but λ inference needs a better model, not more CAISO data); R4 (pricing) confirmed the single-retailer symmetry prediction, and its two open questions — multi-agent reciprocity and item-level Edgeworth — need cross-CHAIN data and a non-subsampled panel that we do not have. Meanwhile every estimator in the library is a point estimator with ad-hoc CIs (profile likelihood, bootstrap, CLT bands), and the papers' next weakness is uncertainty quantification on λ, α, and ℛ jointly.
- **Decision**: Start the Bayesian engine (unit `estimate.bayes`): posterior over (λ, payoff-scale) given choice frequencies via the existing solver + JAX (NUTS or dense-grid posterior for small games), with model-comparison machinery (the F-0008 model rejection becomes a Bayes-factor statement), and the λ-estimator family's disagreement diagnostic re-expressed as posterior conflict. Deeper empirics resume when the PI sources cross-chain retail or finer electricity data; the loaders are ready.
- **Consequences**: p1 gains a calibrated-uncertainty section; the app's Estimate panel can show posteriors instead of point reads; no new data dependencies; the R4/R2 chase items stay parked and explicit in findings.
- **Date**: 2026-08-12

## ADR-0013 — Papers consolidate to p1 + p3; p2/p4 stubs folded

- **Context**: papers/p2_reciprocity and p4_empirical are empty directories from the original four-paper plan. The reciprocity theory and its empirical read (F-0011) live naturally in p1 (instruments — the measurement is the instrument's payoff); the empirical programme's results are p1/p3 sections (F-0008/9 data in p3), and a standalone empirical paper is compelling only when cross-chain retail or finer electricity data lands (both parked chase items awaiting data the PI would source).
- **Decision**: fold p2 into p1 and p4 into p1+p3; delete the empty stubs. Revisit a dedicated empirical paper when new data arrives (the trigger is data, not writing time).
- **Consequences**: two dense papers instead of four thin ones; p1 v0.2 (10pp) and p3 v0.2 (10pp) are the arXiv-track artifacts.
- **Date**: 2026-08-12

## ADR-0014 — R11 merged to main un-gated, by explicit operator direction
- **Context**: `science.plane` (R11) produced the plane-robustness kill-shots. Red team returned **WITHHELD** (criterion substitution O-1, missing CI on the deciding statistic O-2, unpaired control arms O-6). Corrections were applied and committed (`f0268f3`, `b1b833d`, `dbdd6ec`), but the unit was **never re-reviewed**, has **no `gates/science.plane.yaml`**, and no `gates/status.json` entry. The standing merge policy is: merge to main only on fully green gates.
- **Decision**: merge anyway, on the operator's explicit instruction, and record the exception here rather than let it pass silently. The alternative offered (re-review, then gate, then merge) was declined in favour of shipping.
- **On main without a gate**: F-0022, `experiments/plane_robustness.py`, `benchmarks/results/plane_robustness.json`, and the superseded registration `plane_finite_size.*`. The headline is **INDETERMINATE on K2-T1**, not a survival; the load-bearing result is the ceiling criterion (worst `ci_high` +0.156 against a 0.35 ceiling, intervals disjoint at every m).
- **Open objections carried, not closed**: O-3 (re-run D1/D2 paired to the main arm; they draw from independent keys, so the confound-control arm compares a 100-game sample against a 200-game sample), O-6 (D1 drives lambda_normalised +40% where the main arm drives it -32%, so the arms bracket the confound rather than controlling it), and the un-run K4 prior-art re-audit. N>2 is untested entirely.
- **Consequences**: the gate suite no longer covers everything on main, so `run_gates.py --check` passing is now a weaker statement than it was. Required follow-up, in order: re-review the corrected unit, write `gates/science.plane.yaml`, register `science.plane.paired_controls` for O-3/O-6, and only then quote R11 outside this repository.
- **Date**: 2026-08-13

## ADR-0015 — Everything on `main` without a gate, named, with a dated plan to close it

- **Context**: ADR-0014 recorded ONE un-gated merge (`science.plane`, R11) as an exception. It
  did not stay an exception. Since it was written, `science.plane.nplayers` (R12, F-0023/F-0024)
  and a large amount of product code landed on `main` with tests but with **no gate file, no
  `gates/status.json` entry, and in most cases no CHANGELOG line**. The gate suite therefore
  stayed green — 26 units, all GREEN — while covering a steadily smaller fraction of what ships.
  That is exactly the failure shape of **F-0018** (23 gates green while every solver call in the
  shipped wheel raised `FileNotFoundError`, because nothing gated the packaging boundary), and it
  is the reason `run_gates.py --check` passing is now a weak statement.
  Being explicit about this is acceptable; being silent about it is not, so this ADR names the
  whole of it.

- **Decision**: (a) write real gate files NOW for the two research units, with their honest —
  **red** — status rather than a green one bought by omission; (b) name every other piece of
  un-gated code on `main` in the register below, with the tier it will be gated at and a date;
  (c) make the register a standing obligation: **any merge to `main` of library, service or app
  code either lands with a gate or adds a line to this ADR on the same day.**

### Register A — research units on `main`, now gated, both RED

| Unit | Gate file (new) | Why it is RED | Finding |
|---|---|---|---|
| `science.plane` (R11) | `gates/science.plane.yaml` | `red_team_signoff: false` — the review returned **WITHHELD** on 2026-08-13 (O-1 criterion substitution, O-2 no interval on the deciding statistic, O-6 unpaired controls); corrections landed (`f0268f3`, `b1b833d`, `dbdd6ec`) but **the unit was never re-reviewed**. Plus one genuinely open objection, O-7: the K4 prior-art re-audit was never run | F-0022 |
| `science.plane.nplayers` (R12) | `gates/science.plane.nplayers.yaml` | `red_team_signoff: false` — **no adversarial review of this unit has taken place at all**. Three self-raised registration defects are recorded as `accepted` with justification rather than repaired by writing a second criteria file, which is the R11 failure mode | F-0023, F-0024 |

Both gates carry the epistemic disclosures on their face: K2-T1 is **INDETERMINATE**, not a
survival; K3 is a diagnostic with no power, not a kill-shot passed with margin; kill-shot A is
**INDETERMINATE** because its registered precondition failed at N ≥ 3; and A-T2's pass is
**vacuous**. Neither unit's numbers may be quoted outside this repository until the reviews
happen (the rule ADR-0014 already stated, now covering two units).

### Register B — product code on `main` with tests but no gate

Every row has tests and none has a gate, an acceptance artifact, or a claims-ledger entry.
None of them makes a research claim, which is why a **product tier** (code + documentation +
one acceptance artifact reproducing a committed number end to end, as `product.toolkit` does)
is the right closure, not a domain/statistical gate.

| On `main` | Where | Tests | Gate to write | By |
|---|---|---|---|---|
| `problems/` API — `Problem → solve() → Solution`, 9 problem types + `Diagnostics` | `packages/strataq/strataq/problems/` | `tests/test_problems.py` | `product.problems` | 2026-08-15 |
| `solve_situation()` / `Situation` / `SituationSolution` | `packages/strataq/strataq/problems/situation.py` | `tests/test_situation.py` | folded into `product.problems` | 2026-08-15 |
| `fit()` — the Bland–Turocy estimation workflow, panel-preserving, LR tests, `by=` heterogeneity | `packages/strataq/strataq/fit.py` | `tests/test_fit.py` | `product.fit` (must include agreement with `pygambit.qre.logit_estimate` on identical data) | 2026-08-15 |
| `diagnose()` — one verdict, refusals as bounds | `packages/strataq/strataq/diagnose.py` | `tests/test_diagnose.py` | `product.diagnose` | 2026-08-16 |
| `viz` — the shared palette and publication figures | `packages/strataq/strataq/viz.py` | `tests/test_viz.py` | folded into `product.diagnose` | 2026-08-16 |
| `repeated/` — automata, folk theorem, Edgeworth cycles | `packages/strataq/strataq/repeated/` | `tests/repeated/` | `theory.repeated` | 2026-08-17 |
| `evolutionary/` — replicator, Moran, β ≡ λ | `packages/strataq/strataq/evolutionary/` | `tests/evolutionary/` | `theory.evolutionary` | 2026-08-17 |
| `extensive/` — trees, Kuhn's theorem, AQRE, backward induction | `packages/strataq/strataq/extensive/` | `tests/extensive/` (incl. four pygambit cross-checks at the `oracle` tolerance 1e-8, **skipped when pygambit is absent** — so CI may be passing a suite whose oracle checks never ran) | `theory.extensive` | 2026-08-17 |
| New API routes for the four new problem types | `services/api/` (commit `b289cc9`) | API tests | extend `api.core` | 2026-08-18 |
| `/demos` — four explorables incl. `/demos/the-plane` | `apps/web/` (commit `da64d40`) | app tests | extend `web.scaffold` | 2026-08-18 |

`services/` and `apps/web` are named here for completeness; they are owned by other agents and
this ADR does not schedule work inside them beyond the gate extension.

### Register C — ledger debt found while writing this ADR

- **`memory/findings.md` was missing F-0018 and F-0024 entirely**, while `memory/claims.md`,
  `memory/PROTOCOL.md`, `docs/ONBOARDING.md` and `docs/PRD.md` all cite them by number. Both are
  appended in this pass as retroactive entries that cite the primary sources they are
  reconstructed from, and both say on their face that they were written after the fact.
- **The CHANGELOG's `[Unreleased]` section documents `/demos` and nothing else from Register B.**
  Entries for the two research units are added in this pass; the Register B entries are owed with
  their gates.

### Consequences

- `gates/status.json` now carries **two RED units by design**. `run_gates.py --check` fails only
  on *regression*, so a new red unit does not fail CI — which is correct here (nothing regressed)
  and is also precisely why a red unit must be visible on the board rather than absent from it.
- The headline "26 units, all GREEN" in `docs/ONBOARDING.md` §4 is superseded: it is now 28 units,
  26 green and 2 red, and the green count no longer describes the shipped surface.
- Until Register B is closed, the honest statement about the library is: **the instrument layer is
  gated; the product surface on top of it is tested but not gated.** Any external communication
  that implies otherwise is wrong.

- **Date**: 2026-08-13
