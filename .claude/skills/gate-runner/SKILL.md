---
name: gate-runner
description: Ralph-loop gate closure for a work unit. Invoke as /gate <unit-id>. Runs the deterministic gate script, dispatches specialists per failure class, escalates to TRIZ on repeated failure, halts to human after 8 iterations. A unit closes on domain validation, not green tests.
---

# Gate runner

Given a unit id `<unit>` with spec `gates/<unit>.yaml`:

```
while not all_gates_green(unit):
    status = run: uv run python gates/run_gates.py <unit>     # deterministic, scripted
    if status.code_failures:        dispatch numerics-engineer or solver-architect (whichever owns the module)
    if status.domain_failures:      dispatch theory-verifier + the relevant specialist (physicist / econometrician / data-engineer)
    if status.statistical_failures: dispatch benchmarker
    if status.doc_failures:         dispatch doc-writer
    if status.adversarial_open:     dispatch red-team          # artefact + claim ONLY — never the reasoning
    if same_failure_twice(status):  dispatch triz-engine       # MANDATORY escalation, not optional
    if iterations > 8:              halt; update SESSION.md; escalate to the PI
commit (conventional, referencing <unit>); update gates/status.json; update docs + dashboard; push
```

## Rules

- The runner script is the only judge of gate state. Never mark a gate green by hand.
- Track failure signatures across iterations (failing gate id + underlying reason). Two iterations with the same signature = the TRIZ trigger. Do not rationalise it away as "almost fixed".
- **Anti-gaming (hooks enforce, you respect):** a domain gate may not be weakened without a `memory/decisions.md` ADR signed off by red-team; deleting or xfail-ing a failing test to close a gate is a hard failure; `no_todo_no_stub` means no TODO, no `pass # later`, no `NotImplementedError` anywhere in a closed unit; every domain-gate artifact must regenerate via `make reproduce` from its fixed seed — non-regenerable = not green.
- `conjectured`-tier units: the gate spec must state what would count as the claim being wrong. Refuse to close the gate if it doesn't.
- On close: CHANGELOG entry, claims-ledger update if the unit's claim tier changed, dashboard regeneration. Dashboard stale = gate not green.
- On halt: SESSION.md records the failure history, the TRIZ output if any, and the precise question for the human.
