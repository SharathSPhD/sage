# SAGE — Strategic Agent Game Engine

Monorepo: **SAGE**. Library: **`strataq`** (PyPI + import). Research framework: **ThermoQRE**. App: **SAGE Labs**.

## The naming rule
`import strataq`, never `import sage` — SageMath owns the `sage` import. Never publish `sage` to PyPI. A hook blocks `import sage` in commits. Never put "pricing" or "thermo" in a core module name: pricing is a domain plugin, ThermoQRE is a research framework, not the engine.

## Working philosophy
Binding: `research/THERMOQRE_PROGRAMME_v3.md` §0.1. Exploratory instrumentation building — anomalies are the product (log in `memory/findings.md`), ship working code over correct proofs, when theory and numerics disagree assume the theory is incomplete, negative results written plainly, limitations stated once. Discovery ethos in the science; **no freedom at all** about tests, gates, types, docs, reproducibility.

## The three engines — never conflate
- `finite/` — finite N-player strategic form: S = blockdiag(λᵢCᵢ), B, resolvent (I − SB)⁻¹, Hodge on Cartesian-product graphs.
- `population/` — continuum of agents, payoff field F(x), potentiality = symmetry of DF, logit equilibrium = Fisk SUE with the Beckmann potential.
- `bayesian/` — types/interim beliefs. **Deferred**; starting it requires an ADR in `memory/decisions.md`.

Shared machinery (solvers, implicit diff, entropy, dynamics) lives in `core/`. Only response operators and decomposition differ by engine.

## The plugin contract
A domain is a plugin iff it is exactly `{oracle, grid, field, loader, learn}` + `ENGINE` tag (see `strataq/core/protocols.py`) and touches **zero** core code. `ConjugateFieldSpec` is mandatory; `NONE` disables the response instruments rather than faking numbers. Domain code importing from another domain, or a domain change touching `core|finite|population`, fails CI and the hook. Needs core machinery ⟹ it's an engine ⟹ ADR first.

## Gate policy
A work unit closes on **domain validation, not green tests**: `gates/<unit>.yaml` with code / domain / statistical / documentation / adversarial sections all green, artifacts regenerable by `make reproduce` from fixed seeds. Weakening a domain gate needs an ADR + red-team sign-off. Deleting or xfail-ing a failing test to close a gate is a hard failure. `no_todo_no_stub` in closed units means exactly that. Same gate failure twice for the same reason ⟹ TRIZ escalation, mandatorily (`.claude/skills/triz-engine`).

## Adversarial review
`red-team` runs at every unit close and **never sees the implementation rationale** — artefact and claim only. Same isolation for `theory-verifier` reviews. Objections are addressed or logged as accepted limitations; unaddressed objections block the gate.

## Git
Worktrees per track (`wt/engine1`, `wt/calibration`, `wt/empirical`, `wt/product`, `wt/science`); branches `track/<unit-id>`; conventional commits referencing unit ids. Merge to `main` only on fully green gates; rebase before merge; merge at least daily per active track and always before ending a session; push after every merge; `main` always green (CI + gates + docs + fast `make reproduce`). A broken merge is reverted, not patched forward. Stage completions tagged `v0.<stage>.0`. Dashboard stale ⟹ gate not green.

## Session protocol
Start: read `SESSION.md`, this file, and the relevant subtree CLAUDE.md; state a plan. End (and before any /compact): update `SESSION.md` — what changed, in flight, next action, open gates. Never leave a gate half-closed unrecorded.

## Compaction policy
Preserve: all API changes and their rationale; all gate statuses; all anomalies (`memory/findings.md`); the current work unit's contract. Summarise exploration briefly.

## Numerics standing rules
float64 always (`jax_enable_x64`). No literal constants in library code — everything from `config/` through typed schemas. Decompose the **normalised** game. All (I − SB) algebra on the **tangent space**. `log_softmax`/`logsumexp` only — never exponentiate raw payoffs.
