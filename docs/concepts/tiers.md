# The theory tiers

Every scientific claim in SAGE carries a confidence tier, maintained in the [claims ledger](https://github.com/SharathSPhD/sage/blob/main/memory/claims.md) and stated in every relevant docstring:

| Tier | Meaning | Examples |
|---|---|---|
| `exact` | Proved identities. Implemented as verified primitives, cited, never claimed as ours. | Gibbs variational principle; log-partition = CGF; Gibbs measure in potential games; the Hodge decomposition |
| `derived` | Proved in-house and/or numerically checked. Ours to claim, positioned honestly. | The resolvent transfer (χ^eq inherits B's symmetry exactly); ℛ as a λ-free observable |
| `conjectured` | Argued but not established. The ledger states what would count as each one being wrong. | Tightness of the cycling → dissipation chain |
| `speculative` | Exploratory framings, kept out of user-facing claims. | Sparse/entmax responses for 9-ending prices |

Tiers move — in both directions — as evidence arrives. The prior-art history is part of the ledger: what we once thought new and turned out to be textbook is recorded, not erased.
