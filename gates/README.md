# gates/ — the gate closure contract

**A work unit is not closed by passing tests. It is closed by demonstrating that the thing it claims to do is actually true in its domain.**

- `<unit-id>.yaml` — one gate spec per unit (schema documented in `schema.yaml`). Five sections: code, domain, statistical, documentation, adversarial.
- `run_gates.py` — the deterministic runner (Template Method: one class per section). The only judge of gate state; humans and agents never mark a gate green by hand.
- `status.json` — machine-readable current state, consumed by CI (regression check), the pre-commit anti-gaming hook, and the progress dashboard.

Anti-gaming rules (enforced by `.claude/hooks/check_tests_and_gates.py` and CI):
- Weakening a domain gate requires an ADR in `memory/decisions.md` signed off by red-team.
- Deleting or xfail-ing a failing test to close a gate is a hard failure.
- `no_todo_no_stub` means exactly that in closed units.
- Domain-gate artifacts must regenerate via `make reproduce` from fixed seeds; non-regenerable = not green.
- `conjectured`-tier units must state in the spec what would count as the claim being wrong.

The loop that drives a unit to green lives in `.claude/skills/gate-runner/SKILL.md` (`/gate <unit-id>`).

**Known limitation (accepted, stage0 red-team O-4/O-6):** the local guards are three-layered — Claude Code hooks (agent edits), pre-commit hooks (git CLI; requires `pre-commit install`, done at repo setup), and CI + branch protection on `main` (the backstop that cannot be skipped). A bare `git commit -n` on a machine without pre-commit installed defers enforcement to CI; that is detection-then-block at the merge boundary, not prevention at the keystroke, and is accepted. Diff-based guards have no baseline on the repo's first commit; the initial commit was itself red-team-reviewed, which is the compensating control.
