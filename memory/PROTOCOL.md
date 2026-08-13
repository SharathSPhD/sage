# Memory and context protocol

Which file a fact belongs in, what may be trusted, and how context is kept from rotting.
This exists because the failure mode here is not forgetting — it is **confidently
remembering something that is no longer true**.

## 1. The tiers

| Tier | File | Holds | Lifetime |
|---|---|---|---|
| **Intent** | `docs/PRD.md` | what we are building and for whom | changes rarely, deliberately |
| **Direction** | `research/DIRECTION_v4.md` | the claim, the agenda, what stops | per programme phase |
| **Decisions** | `memory/decisions.md` | ADRs — choices and their costs | append-only, never edited |
| **Claims** | `memory/claims.md` | what we assert, with a confidence tier | edited only on tier change |
| **Anomalies** | `memory/findings.md` | F-numbers: what was observed, incl. retractions | append-only |
| **Working state** | `SESSION.md` | what changed, in flight, next action | rewritten every session |
| **Conventions** | `CLAUDE.md` | rules an agent must obey | changes rarely |
| **Derived** | `docs/context-graph.json` | queryable map, extracted from the repo | regenerated, never hand-edited |

**Rule:** a fact lives in exactly one tier. Duplicating it into a second is how they drift
apart. Reference across tiers by id (`F-0022`, `ADR-0014`), never by restating.

## 2. Append-only means append-only

`findings.md` and `decisions.md` are the audit trail. A finding that turns out wrong gets a
**correction appended**, with the original left in place — F-0017 carries two corrections,
F-0019 was partly retracted by F-0020, F-0022 was corrected by F-0024. Deleting a wrong
finding destroys the only evidence that the process works.

## 3. Trust discipline

**Every summary is a claim to verify, not a fact to inherit** — including
`docs/ONBOARDING.md`, including this file. Six specialist agents produced several confident
errors when ONBOARDING was assembled. Before relying on a summary:

- prefer the **extracted** artifact (`context_graph.py`, `gates/status.json`) over prose;
- prefer the **artifact JSON** over any document describing it;
- if a document and the code disagree, the code is right and the document is a bug.

Anything inherited rather than checked must say so in the sentence that uses it.

## 4. Session boundaries

**Start:** read `docs/PRD.md` §0, `CLAUDE.md`, `SESSION.md` tail, and the relevant subtree
`CLAUDE.md`. State a plan before touching anything.

**End, and before any compaction:** update `SESSION.md` — what changed, what is in flight,
the next action, open gates. Never leave a gate half-closed unrecorded.

## 5. What survives compaction

Preserve, always: API changes and why; gate statuses; every anomaly; the current unit's
contract; and any **open objection** from a red-team review. Summarise exploration freely —
it is cheap to redo and expensive to carry.

Discard: tool output already written to an artifact; superseded plans; anything regenerable
by `make reproduce`.

## 6. Staleness triggers

Re-verify rather than trust when: a document cites a finding number lower than the highest in
`findings.md`; `gates/status.json` is older than the newest artifact; `SESSION.md`'s next
action is already done; the context graph predates the newest unit; or a claim is quoted
without the scope its finding attached to it (λ, m, **N**).

## 7. The recurring lesson

A check that passes tells you only that the check passed. Before trusting one, ask what it
would have to see to fail, and whether it could see it.
