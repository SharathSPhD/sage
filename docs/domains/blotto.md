# Blotto — the α > 0 anchor

**Engine**: finite. **Conjugate field**: battlefield budgets (experimenter-set, exactly observable). **Loader**: none — synthetic-only, deliberately: round-level experimental data availability is inconsistent (Chowdhury–Kovenock–Sheremeta; Arad–Rubinstein; Duffy–Matros), so no empirics are promised until data is in hand (DOMAINS v1 §4.2).

**Why it anchors the bracket.** Payoffs are known *by construction* — no demand system, no estimation contaminating the instrument reading. Together with congestion (α = 0, population engine), Blotto brackets the α axis with free payoffs: the same meters that must read zero on congestion must read high here.

**Setup.** Two players allocate integer budgets over k battlefields; larger allocation wins a field (ties split); payoff = value of fields won. The action grid is the lattice simplex of allocations — $\binom{B+k-1}{k-1}$ actions, which is where the matrix-free path will earn its keep at scale.

**Measured** (gate `domains.blotto`, artifact `blotto_readings.json`, regenerable):

| Instance | α | ℛ | EPR |
|---|---|---|---|
| symmetric, B=3, k=3 (10×10) | > 0.6 | > 0.1 | — |
| asymmetric values (2,1), B=2, k=2 | — | — | > 10⁻⁴, circulating |
| **degenerate**: equal values, B=2, k=2 | — | — | **exactly 0** (constant payoffs — every profile ties; a correct null, kept as a regression test) |

The budget field is live: moving B changes the grid and the readings — the perturbation the response instruments will differentiate against in the α > 0 half of the calibration suite.

**Limitations, stated once.** Small instances only on the dense path (the profile space is the product of two allocation simplices); Roberson-style continuous Blotto equilibrium structure is not claimed — these are logit-QRE readings on discrete grids.
