---
name: triz-engine
description: Contradiction resolution by TRIZ. Triggered automatically when a gate fails twice for the same underlying reason, when red-team finds a contradiction where fixing one side breaks the other, or when any requirement pair appears mutually exclusive. Not optional when triggered.
---

# TRIZ engine

## Method

1. **Formulate** the contradiction precisely as *"improving X degrades Y"*. Name X and Y as measurable parameters, not vibes. If you cannot phrase it this way, it is not yet a contradiction — go back and isolate it.
2. **Classify**:
   - **Technical** — two different parameters conflict (speed vs conditioning).
   - **Physical** — one parameter must take two opposite values (the game must be decomposed exactly AND must never be materialised).
3. **Physical ⟹ separation.** Try, in order: separation **in time** (different phases of computation), **in space** (different regions of parameter/state space), **upon condition** (declared capability switches behaviour), **between whole and part / scale** (exact at one scale, structural shortcut at another). This resolves most of them.
4. **Technical ⟹ matrix.** Map X, Y onto the parameters in `references/matrix.md`, look up candidate principles, read their software/maths adaptations in `references/principles.md`. The `triz-knowledge` MCP server (lookup_matrix, get_principle, get_separation_principles) is available as a backing knowledge base for the classical matrix.
5. **Ideal Final Result.** State what the system looks like if the contradiction simply does not arise — then work backwards to the smallest structural change that gets there.
6. **Output**: 3–5 candidate resolutions, ranked, each with (principle invoked, mechanism, cost, what it forecloses). Record the adopted one in `memory/decisions.md` as an ADR referencing the triggering gate/objection.

## Calibration examples (real, resolved — match this standard)

- **Exactness vs scale.** Exact Hodge needs the full payoff tensor; large grids make that infeasible. → *Separation by scale* (+ Principle 1, Segmentation / Principle 17, Another dimension): the game graph is a Cartesian product of complete graphs, so the Laplacian is a Kronecker sum with closed-form eigenbasis — an exact separable transform at near-linear cost. (PROGRAMME v3 §1.2 — the canonical example: the "contradiction" dissolved by exploiting structure, not by approximating.)
- **Generality vs domain validity.** A generic engine can't validate domain-specific claims; a domain-specific engine doesn't generalise. → *Separation upon condition* (+ Principle 23, Feedback / Principle 3, Local quality): `ConjugateFieldSpec` declares per-domain what is measurable; the engine refuses what the domain can't support.
- **Speed vs conditioning near criticality.** Fast iterative solvers degrade exactly where the science is interesting. → *Separation in space* (+ Principle 15, Dynamics): matrix-free GMRES away from criticality, dense tangent-space eigendecomposition near it, switched on `distance_to_criticality`.

A good resolution is **structural** (the conflict stops existing) rather than a compromise (both sides get less). If all candidates are compromises, say so and recommend the least bad with its cost stated.
