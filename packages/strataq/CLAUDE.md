# strataq — library conventions

- **Protocols, not ABCs.** Structural typing (`typing.Protocol`) for `PayoffOracle`, `Solver`, `ResponseOperator`, `Estimator`, `Loader`. No inheritance hierarchies.
- **Equinox modules**, frozen dataclasses; immutability by default — no in-place mutation of arrays or state.
- **Functional core / imperative shell.** Anything JIT-able must be pure; I/O and orchestration stay outside.
- **Registry** (decorator-based) for domains, solvers, estimators, engines — config-selectable by string.
- **float64 always**; the package `__init__` enables x64 and nothing may undo it.
- **No literal constants** — tolerances, seeds, grid sizes, λ ranges come from `config/` via `strataq.core.config` typed schemas.
- **Docstrings**: NumPy style, with a `References` section citing the source result AND its confidence tier (`exact` / `derived` / `conjectured` / `speculative`) from `memory/claims.md`. Every named quantity in PROGRAMME v3 §3 becomes a named, tested function.
- **Tests per module, all four kinds**: exact-identity (1e-10..1e-12), property (hypothesis), oracle (pygambit / analytic potentials), regression (golden outputs, fixed seeds, `tests/golden/`). Coverage floor 90% on core/finite/population/thermo. TDD: red → green → refactor, always.
- **Matrix-free** where B is large: Lineax GMRES on v ↦ v − S(Bv), B never materialised.
- Write modules fully or edit with targeted replacements — never assume unseen code exists.
