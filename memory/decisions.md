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
