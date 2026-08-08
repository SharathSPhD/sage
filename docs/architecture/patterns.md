# Design patterns — the ones that carry weight here

Each is used deliberately; if you find yourself fighting one, the fight goes to an ADR, not a workaround.

## Protocol / structural typing
`PayoffOracle`, `Solver`, `ResponseOperator`, `Estimator`, `Loader` are `typing.Protocol`s (`strataq/core/protocols.py`). Any object with the right shape conforms — no ABC inheritance hierarchies, no registration ceremony to *implement* (only to *select*). Chosen because domains and oracles come from wildly different worlds (a PyTorch demand model, a BPR formula, a dispatch stack) and forcing a base class on them creates false coupling.

## Registry
Domains, solvers, estimators and engines register by decorator and are selected by string from config. This is what makes `method="mirror"` in a YAML file reach the right implementation without an if-ladder.

## Strategy
Solvers (`damped`, `anderson`, `mirror`, `homotopy`) are interchangeable behind one interface; the phase of the problem (near/far from criticality) picks the strategy, per `config/engines/finite.yaml`.

## Adapter
External payoff models (DreamPrice, pyblp, BPR, ERCOT dispatch) are adapted to `PayoffOracle`. The library never learns their internals; ports are validated against the original (the DreamPrice JAX port asserts 1e-6 agreement against the Torch path in CI).

## Builder
`ActionGridBuilder` turns continuous decision spaces into discrete action grids — configuration in, grids out, so grid construction choices (bounds, resolution, empirical support) are inspectable objects rather than scattered arguments.

## Facade
`strataq.__init__` exposes ~15 functions; everything else is a subpackage import. The facade grows only as gates close.

## Functional core / imperative shell
Pure JAX transformations inside (everything JIT-able is pure, `equinox.Module` frozen, no in-place mutation); I/O, config, orchestration outside. This is what makes `vmap`/`grad`/implicit-diff composition safe.

## Template Method
The gate runner (`gates/run_gates.py`): one section class per gate section, fixed skeleton, deterministic verdicts.

## Repository
Data loaders present one interface (`load` + `validate`) over HF, TNTP, ERCOT and local files, with dataset gotchas encoded in the loader rather than in analysis code.
