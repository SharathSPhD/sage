---
name: adversarial-review
description: Run red-team review on a unit or stage. Invoke as /adversarial-review <unit-id | stage-N>. Enforces the isolation rule and the objection-disposition cycle.
---

# Adversarial review

1. **Assemble the package for red-team — artefact and claim only.** The unit's code/doc/benchmark artefacts, the gate spec's `claim:` line, and the relevant claims-ledger entries. NEVER include: the implementing conversation, design rationale, commit messages beyond one-line subjects, or this session's reasoning. If you cannot assemble a rationale-free package, stop and say so.
2. Dispatch the `red-team` agent with the package. For stage boundaries, request the full hostile-referee report and file it in `papers/reviews/stage-<N>.md`.
3. Receive the numbered objection list. For each objection, drive a disposition:
   - **Addressed** — a concrete code/doc change plus a one-line response, recorded in the gate file under `adversarial.objections`.
   - **Accepted** — logged as a limitation in the *one* right place (docs page or paper section), with the gate file noting where.
4. Re-dispatch red-team to verify dispositions (again artefact-only). Sign-off recorded in `gates/<unit>.yaml` → `red_team_signoff: true` only by red-team's explicit statement.
5. Objections that keep bouncing (two rounds unresolved because fixing one side breaks another) trigger the TRIZ skill — that is the standing rule, not a choice.
