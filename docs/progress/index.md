# Progress

*Generated 2026-08-08T18:09:11+00:00 from gate state of 2026-08-08T18:09:11+00:00. Regenerated on every merge — if this page is stale, the gate is not green.*

=== "Plain language"

    **What this project is doing.** Building measuring instruments for strategic systems — how sharply players respond to incentives (<abbr title="logit precision: how strongly payoff differences translate into choice probabilities">λ</abbr>), how far a system is from the "well-behaved" regime where everything settles down (<abbr title="harmonic fraction of the **normalised** game: ‖u^H‖/(‖u^P‖+‖u^H‖) ∈ [0,1] from the Candogan flow decomposition">α</abbr>), and whether give-and-take between players is balanced (<abbr title="‖χ^eq − χ^eqᵀ‖_F / ‖χ^eq + χ^eqᵀ‖_F">ℛ</abbr>) — then pointing those instruments at road networks, pricing data, electricity markets and game experiments.

    **What works now.** 1 work unit(s) fully closed (every closure includes an adversarial review by a hostile reviewer who never sees the authors' reasoning). Claim ledger: 8 established results implemented, 4 results of our own, 2 open conjectures each with a stated way to be proven wrong.

    **The map.** Each dot is a system we point the instruments at; green means that system's anchor is in place.

    <svg viewBox="0 0 700 110" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="alpha axis with domain anchors"><line x1="40" y1="60" x2="660" y2="60" stroke="currentColor" stroke-width="2"/><text x="40" y="95" font-size="13" fill="currentColor">α = 0 (potential — everything known)</text><text x="660" y="95" font-size="13" text-anchor="end" fill="currentColor">α = 1 (harmonic — cycles live here)</text><circle cx="40" cy="60" r="7" fill="#9e9e9e"/><text x="40" y="40" font-size="12" text-anchor="middle" fill="currentColor">congestion</text><circle cx="257" cy="60" r="7" fill="#9e9e9e"/><text x="257" y="40" font-size="12" text-anchor="middle" fill="currentColor">pricing</text><circle cx="350" cy="60" r="7" fill="#9e9e9e"/><text x="350" y="40" font-size="12" text-anchor="middle" fill="currentColor">electricity</text><circle cx="598" cy="60" r="7" fill="#9e9e9e"/><text x="598" y="40" font-size="12" text-anchor="middle" fill="currentColor">blotto</text><circle cx="660" cy="60" r="7" fill="#9e9e9e"/><text x="660" y="40" font-size="12" text-anchor="middle" fill="currentColor">RPS</text></svg>

    **Track health**

    | Track | State |
    |---|---|
    | A · Engine 1 core | not started |
    | B · Calibration | not started |
    | C · Empirics | not started |
    | D · Product | not started |
    | Foundation | healthy — 1 unit(s) closed |

    **What's next.** The first new artefact in the world: the reciprocity meter reading exactly zero on a road-congestion game and clearly positive on rock-paper-scissors — the same measurement, two systems, opposite readings. Everything else follows from that working.

    **Anomalies logged:** 0 (anomalies are the product — each gets chased).

=== "Technical"

    **Gate matrix** (a unit closes on domain validation, not green tests)

    | Unit | code | domain | statistical | documentation | adversarial | Overall |
    |---|---|---|---|---|---|---|
    | `stage0` | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 |

    **Claims ledger** — 8 `exact` · 4 `derived` · 2 `conjectured` · 2 `speculative` — [full ledger](https://github.com/SharathSPhD/sage/blob/main/memory/claims.md)

    **Benchmark results**

    _No benchmark results yet — instruments arrive in Stage 1._

    **Open red-team objections**

    _None open._

    **Gate flow** — what "closed" means

    ```mermaid
    flowchart LR
        W[work unit] --> C{code gates}
        C -->|tests, types, lint, coverage, no stubs| D{domain gates}
        D -->|claim true in its domain, artifacts regenerable| S{statistical gates}
        S -->|effect sizes, CIs, n justified, seeds| Doc{documentation}
        Doc --> A{adversarial}
        A -->|red-team sign-off, objections dispositioned| G[merge to main + dashboard refresh]
        A -->|same failure twice| T[TRIZ escalation]
        T --> C
    ```
