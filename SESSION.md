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
