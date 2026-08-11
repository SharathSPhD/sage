# SESSION.md — rolling session log

## Protocol

**Session start**: read this file, root `CLAUDE.md`, and the relevant subtree `CLAUDE.md`; state a plan.
**Session end (and before any /compact)**: update this file — what changed, what is in flight, the next action, any open gate. Never end a session leaving a gate half-closed without recording it.

---

## 2026-08-08 — Stage 0: Foundation

**What changed** (all of it — this session created the repo):

- Monorepo scaffolded at `~/projects/sage`, wired to `github.com/SharathSPhD/sage`; research docs copied to `research/` (+ `research1.md` conversion). ADRs 0001–0007 in `memory/decisions.md`.
- uv workspace: `strataq` (protocols + config schemas real and tested), `strataq-client` (stub), `strataq-bench` (BenchmarkResult schema real and tested). 25 tests green; ruff + mypy strict clean.
- Plugin contract implemented: `strataq/core/protocols.py` (PayoffOracle, ActionGridBuilder, ConjugateFieldSpec + NONE sentinel, DatasetLoader, LearnPageSpec, DomainPlugin).
- `config/` Hydra tree + typed schemas (`core/config.py`); tolerance ladder; seed policy.
- CLAUDE.md hierarchy (root + 9 subtree files); 13 subagents in `.claude/agents/`; 6 skills in `.claude/skills/` (gate-runner, triz-engine + matrix/principles references, adversarial-review, experiment-runner, docs-sync, release).
- Hooks live: import-sage guard (PreToolUse + pre-commit), plugin-boundary, anti-gaming (test deletion/xfail/gate regression need `SAGE_ADR_REF`), secret scan, ruff/mypy-on-edit. All hook scripts tested.
- Gates machinery: `gates/run_gates.py` (deterministic, Template Method), `schema.yaml`, `stage0.yaml`; `make reproduce` / `reproduce-fast`.
- Memory seeded: claims ledger (K1–K8 exact; R1, N1, N2, N3 derived; C1–C2 conjectured with falsifiers; S1–S2 speculative), literature (+ full nearest-live-work report), glossary, findings (empty), open questions (10).
- **Prior-art sweep done** (theory-verifier): arXiv:2405.07224 is orthogonal (continuous-time replicator convergence/recurrence vs our discrete-time response/thermo observables). N1/N2 not found in literature; N3 upgraded to derived (ADR-0007). No collaboration needed; cite as SOTA; their open discrete-time question is our territory.

**In flight**: nothing — Stage 0 closed this session.

**Adversarial record**: red-team raised 6 objections on the scaffold (O-1 dynamic-import bypass, O-2 blanket ADR override, O-3 no gate baseline — all blocking; O-4/O-6 accepted limitations; O-5 test gap). O-1/O-2/O-3/O-5 addressed and verified on a second cold pass; O-4/O-6 documented once in gates/README.md. Sign-off granted; dispositions recorded in gates/stage0.yaml.

**Next action**: finish Stage 0 close (task list in session), then open `wt/engine1` (Stage 1 Track A: core/types → games → solvers) and `wt/product` (Stage 4 Track D: Learn scaffold). First Stage 1 milestone: `reciprocity_defect()` reading 0 on a congestion game and > 0 on RPS, gate green including adversarial sign-off, visible on the dashboard.

**Open gates**: none. `stage0` — **GREEN** (all five sections, red-team sign-off granted 2026-08-08).

---

## 2026-08-08 (cont.) — Stage 1 Track A: first instruments closed

**What changed** (branch track/finite.core → main):
- `finite.core`: DenseTensorGame, normalise, verified library (Rosenthal congestion + explicit potential, coordination, common-interest, RPS family, matching pennies), JIT damped logit-QRE solver. Exact identities verified to 1e-12; Gibbs correspondence on congestion to 1e-10.
- `finite.response.reciprocity` — **GATE GREEN, the programme's first milestone**: tangent-space machinery (Helmert), S/B operators, chi_partial (K7), chi_equilibrium (Result 1), reciprocity_defect (Result 2), strategic_spectrum + critical_lambda. Five regenerable artifacts: ℛ ≤ 9e-17 on 5 potential games; ℛ ∈ [0.43, 1.2] harmonic; Spearman ρ(ℛ,α) = 0.982 (n = 2000, λ fixed at 1.2, bootstrap CI); χ–FD agreement 1.3e-8; SB spectrum exactly real on potential games across λ ∈ {0.5..10} (N3 leg).
- `finite.decompose.hodge`: separable Kronecker subset transform, m-weighted Candogan projection, alpha(), make_family() with exact target α; weighted orthogonality + equivalence-invariance tested.
- Red-team round: O-1 (blocking, "λ-free" overclaim) → docs/ledger corrected, findings F-0002; O-2 → sweep-holds-λ-fixed made explicit; O-3 → spectrum_reality artifact added. Verification pass granted sign-off.
- Findings: F-0001 (ℛ unbounded above; MP reads 1.2), F-0002 (ℛ magnitude ∝ λ at small λ; only the zero test is λ-free).

**Next action**: Stage 1 remaining units — homotopy/branch solver + pygambit validation, mirror descent, implicit diff, then core/dynamics + thermo (Glauber, currents, EPR). Stage 4 Track D (Learn scaffold) still unstarted in wt/product.

**Open gates**: none.

---

## 2026-08-08 (cont.) — dynamics.exact closed; first genuine discovery (F-0004)

- **Gate dynamics.exact GREEN** (all five sections): Glauber generator, stationary π, currents, exact EPR. Calibration: π = e^{λΦ}/Z to 2.5e-16 on congestion; EPR/J* < 1e-12 potential; NESS on RPS/MP.
- **F-0004 — the programme's first discovery, found by red-team stratification**: marginal ρ(EPR, ℛ) = 0.993 is α-confounded; within-level coupling is +0.80..0.88 for α ≤ 0.65 but **reverses to −0.355 at α = 0.95**. C1's falsifier partially realised; ledger refined; working hypothesis (ℛ's symmetric denominator underflow at α→1) logged for chase. p3_noneq material.
- Red-team round: O-1/O-2 (stratified honesty) addressed with permanent per-level artifact metrics; O-4 floor param removed. Verification granted.

**Next**: homotopy/branch + pygambit validation (solve.branch unit), mirror descent, implicit diff; then trajectory estimators (thermo.estimators); population engine + TNTP (Stage 2 bracket); Learn scaffold (Track D).

---

## 2026-08-08 (cont.) — solve.advanced closed

- **Gate solve.advanced GREEN** (zero red-team objections — first clean pass): magnetic mirror descent (last-iterate; agrees with damped to 7e-13), implicit differentiation via custom VJP sharing the (I−SB) resolvent (Jacobian ≡ χ^eq to machine epsilon, verified at h≠0 too), pygambit cross-validation (1.2e-9 over 32 games × 2 λ, uniqueness confirmed via ρ(SB) < 1).
- pygambit 16.7 installed (dev dep); label-based outcome API quirk documented in validate.py.

**Next**: homotopy/branch tracer (solve.branch) OR open Stage 2 calibration bracket (population engine + TNTP, blotto plugin) and Stage 4 Learn scaffold. Priority per DOMAINS v1: the calibration bracket.

---

## 2026-08-08 (cont.) — domains.blotto closed: first plugin through the contract

- **Gate domains.blotto GREEN** (red-team: zero objections): BlottoOracle + allocation grid + budgets-as-ConjugateFieldSpec + LearnPageSpec, loader honestly None. α = 0.69 / ℛ = 0.12 (budget-3 symmetric), EPR = 0.098 on asymmetric 2-field, **exact zero** on the degenerate constant-payoff instance (kept as regression).
- **F-0005**: Blotto is mixed (α ≈ 0.7), not pure harmonic — zero-sum ⇏ harmonic-pure. RPS stays the α = 1 extreme; Blotto is the realistic high-α anchor.
- Boundary hook verified live on real plugin files.

**Stage state**: Stage 1 units all green (core, hodge, reciprocity, dynamics, solve.advanced). Stage 2 opened with the α > 0 anchor. **Next**: population engine + Fisk/Beckmann + TNTP loader (unit population.core + domains.congestion — the α = 0 anchor with real network data), then homotopy branch tracer, thermo estimators, Learn scaffold (Track D in wt/product).

---

## 2026-08-08 (cont.) — THE CALIBRATION BRACKET IS CLOSED

- **Gates population.core + domains.congestion GREEN** (combined red-team review, zero objections): Engine 2 (routing population games, Fisk SUE via damped Newton on the convex Beckmann-entropy program — the damped logit map two-cycles on Braess, documented), DF-symmetry potentiality, toll susceptibility (χ vs FD 2.8e-9, sign verified by re-solve), TNTP loader (Sioux Falls fetch/cache, offline-safe parser tests), k-shortest route sets.
- **Readings on real Sioux Falls data: ℛ = 5.7e-17, DF defect = 0, KKT = 7e-15.** The meters read exactly zero where theory says zero, on real network data. UE link-flow gap 0.73 explicitly diagnostic-only (route-set/OD-subset restriction documented).
- ADR-0008: boundary rule refined — plugins may import their own declared engine (hook reads ENGINE declaration; cross-engine and cross-domain still blocked, both directions tested).
- **The α axis now has anchors at both ends with free payoffs**: congestion (α = 0, real data, ℛ = 0) and Blotto (α ≈ 0.7 synthetic, ℛ > 0.1), with RPS as the α = 1 extreme. DOMAINS v1 §8's "much stronger position" reached.

**Next**: Learn scaffold + dashboard visuals (Track D), homotopy branch tracer, thermo trajectory estimators, then Stage 3 (pricing/ERCOT) and the α×λ phase map (Stage 5).

---

## 2026-08-08 (cont.) — science.phase_map closed: the money figure exists

- **Gate science.phase_map GREEN** (red-team two-round: wedge-onset wording precision + supercritical-R caveat enforced): 9λ × 11α × 5-game median surface of ρ(SB), ℛ, EPR + supercritical fraction; SVG on the dashboard.
- **F-0006 — criticality escape**: on potential games ρ(SB) is non-monotone in λ (peak 0.51 at λ=1.7, → 0 at λ=15): high-λ QRE concentrates on strict equilibria, C → 0 kills S = λC. Mechanism verified by red-team (σ 99.99999% pure, converged). The supercritical wedge: median game crosses ρ=1 from α=0.5 (0.2 game-fraction at 0.4), frontier descends to λ_c ≈ 3 by α=0.8. F-0004's decoupling regime sits inside the wedge — connection flagged for chase.
- ADR-0009: boundary-hook ADR escape hatch now also reads the HEAD commit message (CI-compatible, auditable) — fixes the CI guards failure on the bracket merge.
- py.typed markers added to strataq/strataq-bench (real packaging gap found by gate mypy).

**Open gates**: none. Nine units green: stage0, finite.core*, finite.decompose.hodge*, finite.response.reciprocity, dynamics.exact, solve.advanced, domains.blotto, population.core, domains.congestion, science.phase_map (*covered under reciprocity gate paths).
**Next**: Learn scaffold (Track D, wt/product), homotopy branch tracer, thermo trajectory estimators, Stage 3 empirics (DreamPrice port, ERCOT).

---

## 2026-08-08 (cont.) — solve.branch closed; session wrap

- **Gate solve.branch GREEN** (red-team: zero correctness objections; two docstring precision nits fixed): pseudo-arclength tracer in tangent-logit coordinates, Newton corrector on the augmented system, fold flags via λ-direction reversal, ρ(SB) along the branch. Fixed-λ agreement 8.6e-13; Gambit agreement 5e-8 through the coordination pitchfork.
- CI fully green on the phase-map push (ADR-0009 verified working in CI).

**Eleven gates green**: stage0, finite.response.reciprocity, dynamics.exact, solve.advanced, domains.blotto, population.core, domains.congestion, science.phase_map, solve.branch (+ finite.core/hodge inside the reciprocity gate). Six findings logged (F-0001..F-0006), two genuinely new (F-0004 decoupling, F-0006 criticality escape).

**Next session start here**: (1) Track D Learn scaffold in wt/product (Next.js; docs/theory 01-10 as single source — none written yet); (2) thermo trajectory estimators (Glauber simulate + KLD/TUR, PROGRAMME §3.5, test 10); (3) Stage 3: DreamPrice JAX port + domains.pricing; ERCOT loader + domains.electricity; (4) λ estimators (estimate/); (5) finer wedge-frontier sweep chasing F-0004/F-0006 connection; (6) performance benchmarks + Kronecker scaling check (open question 5).

---

## 2026-08-08 (cont.) — discovery chase + artifact suite (CINS-parity leg)

- **science.decoupling GREEN** (two-round red-team incl. epistemic disclosures): **F-0007** — H2 confirmed (ℛ denominator-driven at high α, ρ = 0.993), H1 REFUTED (numerator also decouples, ρ = −0.37, CI excl. 0). The response and dissipation layers are distinct observables (interpretation flagged as argued-not-proved). Refuted repair recorded plainly; gate discloses its post-hoc regression-contract nature on its face.
- **Paper p1_instruments**: 9-page manuscript (papers/p1_instruments/p1_main.pdf) — instruments + tiers, calibration table, three findings incl. the refutation, software/reproducibility, honest positioning; figures regenerated from benchmark JSONs (make_figures.py); results_digest.md maps every number to its artifact. Red-team spot-checked 5/5 numbers, abstract bounds, novelty claims, and no-empirical-claims rule: all held.
- **api.core GREEN**: FastAPI surface (solve/decompose/response/dynamics/branch) with provenance + honesty warnings, size/NaN guards; red-team found and we fixed an empty-payload 500 (now 422 from the library boundary); parity to machine eps through HTTP; Dockerfile + render.yaml ready (deployment awaits provisioning approval).
- CINS reference reviewed (github.com/SharathSPhD/CINS): paper+digest / app / site / gates pattern matched and extended (adversarial record, claim tiers, findings log are beyond-CINS elements).

**Open gates**: none — twelve green. **Next**: Next.js app (Learn 01–10 authored in docs/theory + Lab against the API), thermo trajectory estimators, Stage 3 (DreamPrice port, ERCOT), deployment provisioning (ask PI), p3_noneq draft seeded from F-0004/F-0006/F-0007.

---

## 2026-08-08 (cont.) — web.scaffold closed: the full artifact suite exists

- **Gate web.scaffold GREEN** (red-team: zero objections — build 9/9 pages, single-source rule, explainer honesty vs ledger, Lab surfaces warnings): Next.js 15 app with Learn (rendering docs/theory 01/07/09 via symlink — 07 carries the measured facts incl. the refuted repair; 09 concedes the one-price objection outright) and a live Lab (three meters + λ slider + reproduce-in-Python funnel). E2E smoke: API served R = 0.866 on RPS through HTTP.
- Gate runner fix: types/lint checks now apply to .py paths only (markdown paths in gate specs no longer break mypy).

**Thirteen gates green. The CINS-parity suite is complete and exceeded**: library (strataq, 100+ tests) · paper (p1_main.pdf + results digest) · API (FastAPI + Docker + render.yaml) · app (Next.js Learn+Lab) · Pages dashboard with live phase map · claim ledger with tier history · findings log F-0001..F-0007 · adversarial record on every unit.

**Next session**: remaining Learn explainers (02-06, 08, 10); deployment provisioning (Render/Vercel — ask PI); thermo trajectory estimators; Stage 3 empirics (DreamPrice JAX port, ERCOT loader, λ estimators); p3_noneq draft from F-0004/6/7; finer wedge-frontier sweep.

---

## 2026-08-10 — Oracle backend provisioning (ADR-0010)

- OCI CLI configured against the PI's tenancy (API key added via Console by the agent through the browser; fingerprint bf:6c:cc:...:e4; region us-ashburn-1; config at ~/.oci/config on the DGX Spark, key never left the machine).
- Network stack created: VCN `sage-vcn` (10.0.0.0/16), IGW + default route, security list 22/80/443, public subnet `sage-public` (OCIDs in ~/.oci/sage-network.txt).
- Instance launch (A1.Flex 2 OCPU/12GB, Ubuntu 24.04 aarch64, 100GB): **Out of host capacity in all 3 Ashburn ADs** — the documented Always Free failure mode. Persistent scripted retry running on the Spark (~/.oci/retry-launch.sh, cycles ADs every ~10 min; instance OCID will land in ~/.oci/sage-instance.txt).
- Repo side complete: deploy/ (compose + Caddyfile + runbook), docs/ops-hosting.md, api-image.yml (linux/arm64 → GHCR on every main push touching the API), ADR-0010. **When capacity lands**: SSH in → docs/ops-hosting.md §4 (docker, swap, iptables) → copy deploy/ to /opt/sage → `docker compose up -d`.

**Next per SESSION.md**: Learn explainers 02–06/08/10, thermo trajectory estimators, Stage 3 empirics, p3_noneq draft.

---

## 2026-08-10 (cont.) — THE API IS LIVE (interim micro box, ADR-0011)

- PAYG declined (£79 card-verification hold); other regions impossible on Free Tier (home region fixed). **Pivot: VM.Standard.E2.1.Micro launched on the FIRST attempt** (different capacity pool from A1) — 150.136.84.2, Ubuntu 24.04 x86, 4 GB swap.
- Bare-metal deploy (1 GB ⟹ no Docker): uv + systemd (MemoryMax=700M) + Caddy :80. **External smoke: /v1/health ok; RPS ℛ = 0.866; coordination EPR = 0, detailed balance true — the instruments are on the internet at €0.**
- Ops reality vs doc: the Oracle iptables REJECT sits at INPUT 5; rules must go above it (hit and fixed live).
- A1 retry monitor still cycling (~10 min); on landing → compose migration. 48h without → Netcup/GCP per PI.

## 2026-08-10 (cont.) — unit thermo.estimators GREEN (13th gate)

- **Trajectory irreversibility estimators** (PROGRAMME §3.5, estimators 1–2; NEEP deferred pending ADR): uniformised Glauber sampler (`core/dynamics/sample.py`, skeleton per-step EP = EPR/Λ exactly), KLD k-th-order Markov and TUR lower bound (`thermo/estimators.py`).
- Validation: KLD ρ = 1.0 vs exact EPR across ten α levels (~1% per level); TUR tightness ~0.97 near α→0 (linear-response saturation) loosening to ~0.6 at α=0.95.
- Two real bugs caught by TDD/red-team and fixed honestly: (1) fixed-jump-count windows suppress Poisson timing fluctuation → "bound" overshot ×1.5; fixed-horizon truncation now mandatory and tested. (2) point estimate straddles exact near saturation even after debiasing (E[J̄²], Jensen-on-1/Var) → certification moved to `tur_epr_bound_ci` bootstrap-lower quantile, exceedance surfaced as artifact metric + effect size.
- Red-team: 1 blocking + 3 must-address + 1 note, all addressed on re-review; signoff granted. Claims K9/K10 added (both known theorems, implemented and verified).

## 2026-08-10 (cont.) — SAGE Labs redesigned, e2e-verified, DEPLOYED to Vercel (unit web.app)

- **Full app rebuild** (instrument-panel design system, no scaffolding left): landing with LIVE meters (RPS ℛ/σ computed on page load by the VM), Lab (game picker, log-λ slider, σ bars, ℛ/ρ(SB) gauges, EPR + detailed-balance badge, per-game α split bar, full ρ(SB)-vs-λ branch trace with criticality line, honesty warnings surfaced as amber flags), interactive α×λ phase heatmap from the committed surface artifact (4 metrics, hover readouts, supercritical outlines), Learn with KaTeX-rendered theory + prev/next.
- **Issues found & fixed by browser e2e**: no CORS on the API (added middleware, redeployed VM); HTTPS→HTTP mixed content on any hosted frontend (solved by /api same-origin proxy rewrites); client env var not NEXT_PUBLIC (earlier); raw $$ LaTeX in Learn (KaTeX build-time pipeline); Next 15.1.6 rejected by Vercel (CVE-2025-66478 → 15.5.23).
- **Deployed**: https://sage-labs.vercel.app (project sage-labs, prod, iad1 — same region as the VM). Verified in-browser: landing live strip, Lab full readings on congestion + RPS across λ, phase map hover, Learn math — all against the live Oracle backend.
- Pending operator clicks: disable Vercel Authentication (Settings → Deployment Protection) to make the URL public; set Root Directory = apps/web on the now-connected GitHub integration so git pushes deploy.

## 2026-08-10 (cont.) — p3_noneq working draft v0.1

- 7-page draft in papers/p3_noneq/: setup (Glauber/NESS + meters), phase map (F-0006: criticality escape + supercritical wedge), decoupling (F-0004/F-0007: within-level sign reversal, H1 refuted/H2 confirmed, structural local-derivative-vs-global-flux reading), estimators (K9/K10: KLD rho=1.0 recovery; TUR fixed-horizon + debiasing + bootstrap certification, tightness-as-diagnostic), discussion with explicit falsifiers, provenance table mapping every claim to its gate artifact. Figures regenerate from committed artifacts only (make_figures.py — no solves in the paper pipeline).
- Vercel: PI disabled deployment protection and set Root Directory — https://sage-labs.vercel.app now PUBLIC (200s anonymous on /, /lab, /phase, /api/v1/health); git pushes auto-deploy.

## 2026-08-10 (cont.) — Plan v2 approved; phase A1 delivered (unit web.explorables)

- **Full-project re-assessment** with the PI: capability audit (40 library capabilities → 6 endpoints → ~5 app readings, 0/21 artifacts visualized, doc-promised panels missing) + landscape research (Evolution of Trust, Distill, Complexity Explorables, GTE). Plan v2 approved: research-first interleaved workstreams; hybrid app compute; ERCOT + Dominick's empirics; infra deferred. Continuous-build rule: every research unit ships an app surface.
- **A1 delivered**: lib/qre.ts client math tethered to library goldens (23 checks incl. all primitives, CI job web-math); API v2 (/v1/response/poke — the ℛ measurement procedure, currents+states, seeded /v1/dynamics/sample with KLD/TUR reads; 17 API tests); six explorable panels inline in their Learn pages — softmax collapse (editable payoffs, λ→∞ Nash), simplex portrait (click/keyboard-seeded trajectories), free-energy dial, poke panel (off-diagonal cross-pairing after e2e caught the RPS player-exchange symmetry trap), two dials, one-price objection with regret readout.
- **Red-team (app UX/honesty brief): granted-conditional; all six conditions closed same-session** — λ slider in PokePanel (λ-free symmetry verdict demonstrable), golden coverage extended to every exported client function, provenance badges now accurate, responsive single-column reflow ≤760px, keyboard operability on the simplex SVG, tether test wired into CI. Verified numerically by red-team: TwoDials' "λ never changes the best price" claim, poke reciprocity at large h, one-price fairness (both sides argmax).
- Deployed via git auto-deploy; panels live on sage-labs.vercel.app.

## 2026-08-11 — unit estimate.lambda GREEN (14th gate; plan-v2 R1)

- **λ-estimator family** (`strataq/estimate/`): frequency MLE (profile-likelihood CI, flat-likelihood guard), autodiff MLE through an unrolled solve (CI inheritance disclosed), χ moment-matching (oracle-anchored), dispersion inversion (flat-entropy guard, bootstrap CI, fast no-bootstrap mode for the API). Agreement protocol with documented precedence: identification warnings suppress the disagreement flag.
- Recovery: median rel. error 2.6% (MLE) / 2.8% (dispersion) / exact (χ) over 20 (game, λ*) cases; λ-mixture data widens the spread ×91 and flags; symmetric-RPS data warns "unidentified" instead of returning a number. Config gains an `estimate` section (two honest flatness knobs after red-team O-1).
- Red-team: granted-conditional (inline-literal threshold; undisclosed CI inheritance); both closed + notes; recorded in the gate.
- **App surface (continuous-build rule)**: /v1/estimate/lambda endpoint (sync-budget subset with pass-through warnings) + the Lab's "guess λ" panel — hidden-λ deal, client-drawn sample, your guess vs the live estimator family vs truth.

## 2026-08-11 (cont.) — A2 dynamics theater live

- Doc 05/06 panel deployed: the joint-profile lattice with π as node mass and the exact J* as animated directed flow (absolute noise floor keeps detailed-balance chains visually still); λ slider; live EPR badge; one-click trajectory sampling showing KLD/TUR converging to the exact meter. Verified on the deployed stack: RPS λ=1.5 exact 2.24 / KLD 2.24 / TUR cert 1.65.
- Next per plan: A3 domain labs (Blotto + Sioux Falls) ∥ R2 ERCOT electricity plugin.

## 2026-08-11 (cont.) — A3 (first half): Sioux Falls network lab live

- /v1/domains/sioux_falls/{network,sue}: lazy-cached RoutingNetwork (top-12 OD, k=3), toll guards; API test asserts the physics (tolling the busiest link reduces its flow). Warm SUE solves 0.7s on the micro (16.8s one-time JAX trace).
- /network page deployed: real TNTP node geometry, flows as width, v/c as colour, click-to-toll (the domain's conjugate field) with total-time delta vs untolled, θ dial, live KKT residual, gate ℛ reading. Verified on sage-labs.vercel.app.
- Remaining in A3: Blotto allocation lab. Next research: R2 ERCOT.

## 2026-08-11 (cont.) — unit domains.electricity GREEN (15th gate; F-0008 revised)

- Red-team WITHHELD on the first pass — a substantive catch: plain-FT surrogates Gaussianize the kurtosis-132 LMP marginal (biased null), and the RTM reading sits 5–6× BELOW its null (not "at-null"). Empirical corrections, not rewording: AAFT nulls added as primary (detection requires exceeding both classes), a null_mismatch_low honesty flag in every artifact, claim scoped to n/bins/embedding, F-0008 rewritten with the retraction of "certified null" on the audit trail.
- Final verdicts: **DAM hourly at-null** (0.0447 inside the AAFT band — consistent with a linear time-reversible process); **RTM 5-min no-detection + mismatch flag** (0.0474 vs AAFT q01 0.239 — no linear process with its spectrum+marginal reproduces the Δ-sign persistence; the anomaly is now a chase item with a candidate mechanism: dispatch/ramp constraints).
- Also: 429-backoff mock test; OASIS terms note; coverage 94%; signoff granted-conditional with permanence conditions live in code.

## 2026-08-11 (cont.) — F-0009 verified, stratified, and shipped to the app

- Red-team numerically verified the Markov null (DB exact to 1e-10; FPR 1/30 at α=.01; robust to seeds/bins/ties/Bonferroni/order-2 leakage); blocking subsample condition resolved by weekly stratification (consistent-sign, concentrated in high-ramp weeks 4–5 at p≤.005; LOWO mapped) with the concentration acknowledged wherever the number is used. RTM re-probed at k=2: still at-null (chase item closed).
- **/markets page live**: the actual July-2026 SP15 series with weekly-ratio shading, the verdict table against the reversible null, and the plain-language explanation — all rendered from the committed `electricity_series.json` artifact (no live dependency). Continuous-build rule satisfied for F-0009.

## 2026-08-11 (cont.) — electricity plugin contract complete; conditional λ pipeline rejects its own model

- BiddingOracle (uniform-price, D=1, ties split) + OfferGridBuilder + exact-linear offer-shift ConjugateFieldSpec + PLUGIN registration — five-object contract satisfied and contract-tested; clearing-price dispersion monotone in λ (the identification channel) unit-tested; auction α confirmed mixed.
- The conditional λ̂ artifact recorded the honest outcome: MODEL REJECTED (dispersion ceiling 10.4 < observed 16.8) — no λ reported; richer supply model queued. Task #37 closed; /markets page live with F-0009.

## 2026-08-11 (cont.) — unit science.frontier GREEN (16th gate; F-0010)

- **Scale folding**: σ(λ, s·u) = σ(sλ, u) checked SHARPLY (identity error 0.0) — the α=0 criticality peak is a pure λ×payoff-scale fold; the sweep's 23% spread labelled honestly as grid coarseness. F-0006 item 1 closed.
- **Frontier**: λ_c(α) = the verified-unique crossing of the fixed-set median-ρ curve (single crossing checked per level), monotone descending 7.8→3.0 over α 0.55→0.80; the 5-game-vs-40-game onset difference recorded as sampling variability.
- **F-0010**: the initial reversal criterion FAILED 2/4 and the failure is the finding — the universal fact is the coupling COLLAPSE at α=0.95 (all λ, both sizes); the sign is λ-dependent across three λ, each within ~2 null-SD. Criterion revision + rationale locked in config.
- Red-team: granted-conditional; all five conditions (identity framing, single-crossing verification, threshold rationale, sign-language, 'pre-registered' wording) closed same-session and recorded in the gate.

## 2026-08-11 (cont.) — p3_noneq v0.2 (8 pages)

- Data section added (four nulls → F-0009 detection ~1.1 nats/day concentrated in scarcity weeks; RTM at-null; model rejection; the retraction in the text); phase section closes both v0.1 open questions (exact fold identity; λ_c curve with single-crossing verification); decoupling section carries F-0010 (universal collapse, λ-dependent sign, failed-criterion honesty); abstract updated to "through to real data".

## 2026-08-11 (cont.) — A4 findings gallery live

- /findings: all ten findings as status-chipped cards — discoveries, corrections, certified nulls, the refuted repair, the retraction, the failed-criterion finding, in equal typography — plus inline charts from committed artifacts (decoupling collapse behind a you-predict-it reveal, the λ_c frontier, estimator tracking) and play-with-it links. Deployed and verified.

## 2026-08-11 (cont.) — p1_instruments v0.2 (10 pages)

- New §"Estimating λ" (R1's four routes + honesty protocol, recovery numbers, ×91 mixture diagnostic, and the first-market-contact model rejection), cross-referencing p3 for the dissipation claims. v0.1 → v0.2.

## 2026-08-11 (cont.) — A5: the 5-minute tour live

- /story: five narrative beats ("landscape or whirlpool?") threading the existing panels — noise dial, the loop, still-vs-turning water, the poke trick, real systems — ending at the Lab; landing hero leads with the tour. App workstream A1–A5 complete (Blotto lab the one optional remainder).

## 2026-08-11 (cont.) — R4: domains.pricing GREEN (17th gate) — F-0011, the reciprocity meter's first empirical read

- HF Dominick's panel (`qbz506/dreamprice-dominicks-cso`, per the PI: use what's on HF) → gap-tolerant loader (margin-window cleaning, cost from gross margin), brand indices, category series; 3 tests.
- **F-0011**: Campbell↔Progresso cross-brand cost pass-through, two-way demeaned, cluster-bootstrapped (re-demeaned per resample): own 1.07/0.97, cross ~0.003/0.0005, asymmetry CI ∋ 0, **ℛ_emp = 0.0011 [0.00005, 0.005]** — the single-retailer symmetry prediction stated in config before the run, confirmed. N2 executed; C2's first empirical anchor. Edgeworth scan 0/30 at-null (lag-1 ac ≈ 0.57, so it's about price dynamics, not basket noise).
- χ row-ordering bug caught by economic sanity inspection pre-review; disclosed in F-0011.
- Red-team granted after one blocking condition (bootstrap re-demeaning — closed by strengthening, CI unchanged); the mechanical-correlation attack on cost=price·(1−margin) CLEARED numerically (residual corr 0.94–0.96, margin CV 0.26–0.35). O-1..O-4 dispositions in the gate.
- App surface: /findings gains the F-0011 card + a predict-then-reveal empirical pass-through matrix panel (committed-artifact provenance badge).

## 2026-08-12 — R5: thermo.protocols GREEN (18th gate) — F-0012, the driving-cost inversion

- `strataq.thermo.protocols`: Hatano–Sasa housekeeping/excess split (σ_hk ≡ 0 iff detailed balance; σ_ex = −dD/dt verified), stepwise λ-quench protocols, Jarzynski + Hatano–Sasa IFTs; 13 tests.
- **F-0012**: excess ⟨Y⟩ collapses across α (0.036 → 2.8×10⁻⁵) while housekeeping grows to 12.4 nats at a constant 0.77 nats/time burn rate; refinement slopes ≈ −1 (the ~1/K law). Pay-per-change vs pay-rent inversion.
- Red-team granted after 3 blocking conditions, all closed by strengthening or honest narrowing: (O-1) the exact-transfer IFT is a telescoping identity — relabelled, and a POWERED sampled check at α=0.5 added (CI [0.995, 1.003] ∋ 1, sampled Y 0.0076 vs exact 0.0068); (O-2) the first-pass mechanism ("λ-insensitive NESS") REFUTED by the red-team's asymmetric-mix probe — mechanism now recorded as OPEN; (O-3) the intended pre-registration commit had ABORTED on a hook failure (config staged, not committed, when results existed) — recorded plainly in F-0012; standing rule upgraded to verify the commit landed via git log before running.
- App surface: /findings gains the F-0012 card + two artifact-badged charts (excess collapse, housekeeping rent).

## 2026-08-12 (cont.) — R6: estimate.bayes GREEN (19th gate) — Bayesian layer + the first EFE auto-research campaign (F-0013)

- TRIZ session (.triz/session.jsonl): principles 10/25/27 → precomputed probe evidence, the solver resolving its own anomaly, cheap disposable probes. IFR 3/4.
- `strataq.estimate.bayes`: grid posterior over λ (self-diagnosing resolution guard, auto-refining `refined_posterior`), scale fold as an exact posterior reparameterisation (error 0.0), mixture-vs-single Bayes factors (decisive ~1e12 on mixture data, Occam 0.11 on clean), and the generic EFE/BALD campaign loop. 10 tests.
- **The estimator's calibration study caught its own bug**: pre-registered coverage FAILED first run (34/50 at λ*=1.8, honestly recorded) → chased to interval quantisation at <6 effective grid points (78% coverage) → resolution bar raised, auto-refinement added → 48/46/48. Red-team's off-grid probe: 27–29/30 at untested λ*.
- **F-0013**: the EFE campaign resolved F-0012's mechanism in ONE probe (α=0.95, EFE 1.19 nats): ness_floor at belief 0.996, validated on all 19 held-out probes (median resid 0.002 dex) and σ-sensitivity re-runs. Red-team condition closed by the 3×3 validation (excess/F flat in α to 0.1%) WITH honest scoping: well-relaxed quenches only — at τ=0.1 the lag dominates and the formula fails; the fast regime stays open.

## 2026-08-12 (cont.) — P1: product.toolkit GREEN (20th gate) — strataq for real users (PI directive)

- `strataq.toolkit`: the three questions in one call each from plain lists — rationality posterior, reciprocity verdict, irreversibility verdict — plus game_thermo; every result carries honesty warnings. `strataq.thermo.nulls`: the decisive reversibilized-Markov null promoted from experiment code into the library.
- Red-team WITHHELD first pass (usage attack): NaN computed through silently, constant series gave a vacuous verdict, noisy χ classified against hard thresholds with no uncertainty. All closed by strengthening: loud validation everywhere, chi_se → Monte-Carlo CI with a verdict that refuses to classify across bands, measured power documented (≥80% at n≥300). 17 tests; acceptance artifact reproduces F-0011's number through the public contract.
- Packaging: README quickstart rewritten for adopters; tag-triggered PyPI trusted-publishing workflow (wheel build verified). Operator action for the PI: create the `strataq` project on pypi.org and add this repo/workflow as a trusted publisher — then `git tag strataq-v0.1.0` publishes.
- Claims ledger gains the Product claims section (P1: facade fidelity, falsifiable).

## 2026-08-12 (cont.) — P1 complete: the product surface is live end to end

- /v1/toolkit/{reciprocity, irreversibility, rationality} deployed to the micro VM (validation as 422s, warnings in every response); live check reproduces F-0011 (r = 0.00113) over HTTP with the point-read honesty verdict.
- /tools ("Your data") live in the app: paste-a-series irreversibility verdict + χ-matrix reciprocity read with optional SEs, defaults set to the real Dominick's estimates; nav updated; Vercel auto-deploys.
- Remaining PI click for full PyPI publication: create the strataq project on pypi.org with this repo/workflow as trusted publisher, then `git tag strataq-v0.1.0 && git push --tags`.

## 2026-08-12 (cont.) — science.quench_regimes GREEN (21st gate) — F-0014, the quench bracket + the guard that caught the machine

- Second EFE campaign (pre-registration commit verified landed): run 1 stopped confidently after ONE probe and the pre-registered held-out guard REFUSED it (winner_failed_validation) — lesson institutionalised as run_campaign's min_probes stopping gate. Run 2 (min_probes=6): gap_interpolation at 0.9999, held-out median 0.089 dex over 24 probes.
- **F-0014**: fast-quench excess bracketed by the path-length floor and the frozen divergence D(π_start‖π_end) — exact AT τ=0 with a NON-UNIFORM approach; single-gap crossover holds on monotone NESS paths; loop-like paths (α=0.95, long ramp) fail by ~1 dex and are reported, not averaged (consumed max 1.06 dex in the artifact after the red-team caught held-out-only reporting).
- Red-team round 2 conditions all closed by correction: 'telescoping identity' language fixed, full residual accounting, min_probes labelled a stopping gate (held-out guard remains the validator), gap-convention choice recorded.
- Docs cookbook shipped (mkdocs nav: "Cookbook (your data)"); p3 §protocols updated with the corrected F-0014 picture (10 pages).

## 2026-08-12 (cont.) — A3 second half: the Blotto allocation lab live

- /v1/domains/blotto/read: full instrument read at chosen budgets (QRE allocation mixes, α, ℛ; EPR when the joint space fits the dense guard, honest warning when it doesn't). 2 endpoint tests; α reproduces F-0005's 0.69 at equal budget 3.
- /blotto in the app: budget sliders (the conjugate field) + λ, live α/ℛ/EPR meters, both colonels' allocation mixes as ranked bars, guided things-to-try. Nav updated. App workstream A1–A5 now fully complete.

## 2026-08-12 (cont.) — science.quench_multimode GREEN (22nd gate) — F-0015, the path-aware quench model + a retraction done right

- Third EFE campaign (pre-registration verified landed; F-0014's failure cells in the grid by construction; full hypothesis × probe residual table in the artifact): **recursive_1mode** — the per-step single-gap recursion tracking p along the ramp — at belief 1.0, held-out validated; dominates the incumbent on median (0.076 vs 0.13 dex) AND worst case (0.44 vs 1.11; loop-path improvement +0.67 dex).
- Red-team round 3 caught two things that matter: (1) run-1's two-mode anti-result was an implementation ARTIFACT (eigenbasis mix-up + truncated lstsq) — retracted on the record; the fixed implementation still loses to one mode, and the lesson is re-attributed to fast-remainder truncation; (2) 2×2-only validation — 3×3 probes added (config addition recorded), claim holds within adequacy there with honestly larger errors.
- p3 §protocols carries the recursion + the retraction note (10 pages).

## 2026-08-12 (cont.) — thermo.hs_estimator OPEN (F-0016): the IFT cannot self-calibrate the plug-in — checkpointed honestly, no gate claimed

- Data-facing HS estimator built + tested (module, sampler, 4 tests green); registered sweep run twice, BOTH failures recorded: covers-1 diagnostic non-monotone (P3 FAIL), equivalence upgrade still false-passes at τ=1 (45% bias behind an IFT ≈ 1.01). Root cause isolated: spectral-gap collapse along the ramp (0.88→0.15) — late windows never settle; the F-0006 concentration effect biting the estimator.
- Unit deliberately left OPEN with next steps recorded in F-0016 (per-window relaxation-time gate from within-window autocorrelation replaces the IFT as primary). ADR-0013: papers fold to p1+p3 (p2/p4 stubs removed).

## 2026-08-12 (cont.) — thermo.hs_estimator: red-team WITHHELD round 2; unit stays OPEN, the failure map is the deliverable (F-0016 final)

- The per-window relaxation gate closed round 1's false-pass regime (monotone boundary, τ=1 fooling case caught, admitted-hold coverage on the registered seed) — then round 2 broke it four measured ways: game-dependent underestimate up to ~19× (α=0), multi-seed coverage ~2/5 at the admitted hold, collapse at small n_trajectories, and the P1 re-scope owned as a criterion weakening (sweep PASS void for certification).
- Response: signoff NOT flipped; module banner-marked EXPERIMENTAL; F-0016 carries the full four-mode failure map + redesign directions (split-sample π̂, explicit bias correction, game-adaptive safety, minimum-n analysis); docs/CHANGELOG corrected to the withheld status. Six tests remain green for what the module does.

## 2026-08-12 (cont.) — hs_estimator: the missing-λ₀-window bug FIXED (bias closed); variance under-coverage remains open

- Root-caused and fixed: the sampler emitted K−1 of K jump windows — with the pre-quench λ₀ window prepended, the estimator is unbiased across seeds (multi-seed mean within 0.01 of exact). Two refuted hypotheses (split-sample π̂, burn-in scaling) on the record first.
- CI moved to a trajectory bootstrap with π̂ re-estimated per resample — still under-covers ~1.7× (common-mode π̂ noise; per-seed coverage 2-3/6). Unit stays OPEN, module banner stays on; the certification path is now a variance model (window-block bootstrap or autocorrelation-corrected delta method) + the game-adaptive safety and minimum-n items.

## 2026-08-12 (cont.) — hs_estimator statistics VINDICATED: 19/20 CI coverage; only the conservative gate flicker remains

- 20-seed study: coverage 19/20 (the bootstrap is calibrated — the earlier 1.7× shortfall was a six-sample mirage), unbiased (0.0911 vs 0.0891). Decomposition shows π̂ noise negligible vs switch-state sampling.
- Remaining before re-registration: relaxation-gate boundary flicker (conservative — 4/20 usable at τ=24), game-adaptive safety, minimum-n. F-0016 updated; unit stays open/bannered until those close.

## 2026-08-12 (cont.) — hs_estimator gates stabilised: 10/10 usable + 10/10 coverage at settled holds

- Relaxation threshold → τ̂ + 2SE (0/10 flicker); IFT companion → anomaly detector (excludes-1), its equivalence form having been pure power-starved flicker once certification moved to the relaxation gate (escalations 3+4 recorded in config). τ=32: 10/10 usable, 10/10 coverage; τ=24 the honest conservative margin (5/10 admitted, all cover). Registered sweep PASSES with a monotone boundary.
- Remaining before re-registration + fresh red-team: game-adaptive safety (α=0), minimum-n analysis. Banner stays on.

## 2026-08-12 (cont.) — hs_estimator: alpha=0 resolved by the gate itself; min-n warning added; ready for re-registration

- α=0 needed no game tuning: the noise-aware gate demands τ ≥ ~86 there (the true basin-hopping timescale), refuses everything shorter, and at τ=110 admits with full coverage (3/4 usable, 3/3 cover). The red-team's ~19x underestimate was the pre-fix machinery on unsettled windows; settled-hold underestimate is ~1.32x, inside safety 4.
- Thin-statistics warning below n=100 added (validated at n ≥ 200). The unit's remaining path: fresh registered config + fresh red-team.

## 2026-08-12 (cont.) — thermo.hs_estimator GREEN (23rd gate): withheld → certified, the full arc

- v2 registration (commit landed pre-run) passed C1–C5 (one recorded C4 probe-point escalation out of the measured marginal zone; calibration 20/20). Fresh red-team's two blocking conditions closed and verified in-repo: the IFT anomaly companion PROVEN to fire on a continuous-ramp non-stepwise input (CI [1.03, 1.10] excludes 1 — permanent test), and the unit/scope contracts made explicit (mandatory λ₀ window, one-time-unit rule with worked example, validated scope 2×2 α∈{0, 0.25} n≥200).
- Verdict GRANTED, zero remaining objections, banner LIFTED. F-0016 closed: two refuted hypotheses, one found-and-fixed bug, five recorded escalations, two red-team rounds — the adversarial machinery converged.

## 2026-08-12 (cont.) — continuous-build surfaces for the 23rd gate

- p3 §protocols gains the data-side quench paragraph (the certification arc in miniature + the gap-collapse consequence: quench dissipation is hardest to measure exactly where strategic systems are most interesting). 10 pages.
- /findings gains the F-0016 card ("The estimator that had to earn it" — withheld → certified).

## 2026-08-12 (cont.) — R7 round 1: WITHHELD, and rightly — the premise itself was the finding

- Red-team caught four foundations failing: the circular protocol half-counted (closure adds +3.7 → loop total 7.0 nats/day), 6 states outside certified scope, iid CI ~28% narrow (day-block bootstrap [6.22, 8.50] now THE interval), and the full-window "admission" was regime-mixing cancellation — their monthly scan (now a permanent artifact metric) shows 5/7 months individually fire the anomaly detector.
- F-0017 REWRITTEN (original in git history): the drift diagnosis leads — the trading day is NOT one repeated quench at monthly granularity; the 7.0 nats/day loop affinity is descriptive only (out-of-scope states; the IFT's open-chain identity doesn't constrain closed cycles). Re-review dispatched on the corrected framing.

## 2026-08-12 (cont.) — strataq 0.1.0 IS ON PyPI; GHCR verified; and the release smoke test caught a showstopper (F-0018)

- PI supplied the PyPI token + GitHub PAT. `pip install strataq` now works: 0.1.0 published, verified from the public index in a clean venv (all four toolkit entry points, F-0011's ℛ = 0.00113 reproduced).
- **F-0018 — the release smoke test earned the whole procedure**: from a built wheel, EVERY solver-touching path failed (`FileNotFoundError`) because the packaged `base.yaml` fallback `base_config()` documents was never shipped; `__version__` had also drifted from pyproject. 23 green gates missed both because every test runs inside the checkout — and the CI wheel smoke checked precisely the one numpy-only entry point that doesn't touch the solver. Fixed, with four release-integrity tests (existence, byte-identity drift guard, standalone schema load, version match) and a CI smoke that now exercises a solver path. Claim P1's falsifier is recorded as REALISED at the packaging boundary and repaired.
- GHCR: `ghcr.io/sharathsphd/sage-api` confirmed public (anonymous pull) and the arm64 image RUNS — health + `/v1/toolkit/reciprocity` both correct in a throwaway container. PAT verified for pushes; `PYPI_API_TOKEN` stored as a GitHub Actions secret so `git tag strataq-v*` publishes from CI. No credential in the repo (`~/.pypirc` 0600 outside it).
