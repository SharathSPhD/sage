# Competitive position — where StratAQ wins, and the changes that make it win

**Audit date:** 2026-08-12. Every number below was verified against the source named.
**Reads with:** `docs/product/PRODUCT_v1.md` (what to build), `research/DIRECTION_v4.md` (the claim).

The instruction behind this document was: *"I had asked to refer to other game-theoretic
libraries/apps like Gambit — that research has not yielded any changes."* This document is
the changes.

---

## 1. The landscape, compressed

| Tool | Scope | Adoption | Last release | What it leaves open |
|---|---|---|---|---|
| **pygambit** | n-player normal/extensive form; **the only production logit-QRE homotopy in any language** | 551★, **2.6k downloads/mo** | 16.7.0, 2026-07-11 | no SEs on λ, no derivative info, no decomposition, no dynamics, not vectorised, **no Linux/macOS wheels** |
| **nashpy** | 2-player only; support/vertex enumeration, Lemke–Howson, replicator | 369★, **34k downloads/mo** | 0.0.43, 2025-11 | no QRE, 2 players. Its own README points n-player users to Gambit |
| **OpenSpiel** | RL-in-games; MMD is the only entropy-regularised solver | 5.4k★, 152k/mo | 2.0.2, 2026-08-12 | MMD = one QRE at one fixed α, 2-player zero-sum sequential only. "QRE" does not appear in the algorithms table |
| **QuantEcon** | `game_theory` incl. **`logitdyn`** | 190k/mo (lecture-driven) | active | ships the Glauber chain and never closes the loop — no logit fixed point, no stationary thermodynamics, no EP |
| **sGameSolver** | homotopy incl. **logit Markov QRE** — the nearest direct competitor | **30 downloads/mo** | 1.0.2, **2023-08** | effectively abandoned; four-call ceremony to one equilibrium |
| **pyblp** | BLP demand | 291★, 1.9k/mo | 1.2.0 | the model to copy, not a competitor |
| **pyRVtest** | conduct testing on pyblp | very small | 0.3.2, 2025-07 | proves "diagnostic layered on a solver, tied to one paper" is a viable form — and that it dies without a runnable first code block |
| **GTE** (web) | Nash in normal/extensive form | 101★ | Flash-era; primary URL **404s**, mirror on a free-tier Firebase subdomain | **no QRE, no λ slider, no branch plot** |
| `time-irreversibility-estimator`, `irreversibility` | irreversibility of a **scalar series** | active, 2025 | — | **no concept of agents, actions, payoffs or profiles.** Cannot say which player's asymmetry produces the entropy |

---

## 2. Five open slots, ranked by how cheaply we can take them

### Slot 1 — pygambit has never shipped a Linux or macOS wheel

Verified against the PyPI release JSON for 16.1.0, 16.2.0, 16.4.0, 16.5.0, 16.6.0 and
16.7.0: **twelve Windows wheels and one sdist, every release. Zero manylinux. Zero macosx.**

Every Mac user, every Linux user and **every Colab notebook** compiles a C++17 codebase from
source to use the field's reference QRE solver. That is the whole of the 13× download gap to
nashpy, which has a fraction of the algorithms and ships pure Python.

**Change:** publish `strataq` wheels for every platform and put *"works in Colab, no
compiler"* in the README's first screen. This is a one-afternoon change and it is the single
highest leverage item in this document. It also fixes a real replication problem: a
replication package that says `pip install pygambit` fails on the referee's Mac.

### Slot 2 — the estimation workflow has a *manual* instead of an API

Gambit's own maintainer co-authored **Bland & Turocy (2025), "Quantal response equilibrium as
a structural model for estimation: the missing manual," *Games and Economic Behavior***. When
the maintainer of the only QRE solver publishes a manual for the estimation workflow, the
workflow is not in the software.

What `pygambit.qre.logit_estimate` returns: `.lam`, `.profile`, `.log_like`. **No standard
error. No confidence interval. No likelihood-ratio test. No bootstrap.** Every published CI
on λ is hand-rolled, and different labs hand-roll it differently — profile likelihood,
numerical Hessian, bootstrap — so the numbers are not comparable across papers.

Worse, the *input* is a `MixedStrategyProfile` of raw counts. Subject identity, round and
treatment are aggregated away **before** the likelihood is written. Any researcher who wants
λ heterogeneity has to write their own likelihood from scratch.

**Change:** `strataq.fit(game, tidy_frame, by="subject").summary()` — tidy data in, panel
structure preserved, λ̂ with a CI, LR against Nash and against uniform, per-group λ, and the
four-estimator agreement protocol (which the repo already has, gated, as `estimate.lambda`)
reported with disagreement flagged rather than averaged. **This is the reference
implementation of the Bland–Turocy manual.** It is the highest-value single deliverable for
the researcher audience.

### Slot 3 — nobody implements the Candogan decomposition

Fifteen years after Candogan, Menache, Ozdaglar & Parrilo (2011), there is **no PyPI package
of any kind** implementing the potential/harmonic/nonstrategic flow decomposition. The only
public artifacts are one-off blog notebooks. The 2026 theory papers that use it as their
baseline coordinate system (arXiv:2605.29919, arXiv:2608.01967) release no code.

`strataq` has it, with the separable Kronecker transform, in near-linear time, gated at
machine precision. **We are the only implementation that exists.**

**Change:** stop burying it. `strataq.alpha(game)` belongs in the README's first screen, and
`plot_decomposition()` belongs in `viz`. Every paper that currently hand-rolls a projection
is a citation.

### Slot 4 — irreversibility tools are strategically blind

Both live packages take a scalar time series and return "is it irreversible?" They have no
concept of an agent, an action, or a profile. They cannot tell you **which player's response
asymmetry generates the entropy production** — which is the only version of the question a
game theorist or a regulator can act on.

Meanwhile Seifert's 2026 *Nature Reviews Physics* review of coarse-grained entropy-production
bounds is the field's canonical reference and **ships no code at all**.

**Change:** `from_trajectories(...)` — accept an observed sequence of action profiles, return
σ_EP with a bootstrap CI against a detailed-balance null, **attributed per player**. The
landscape audit's conclusion was that this single entry point is the one call all six adjacent
communities can make. It already exists inside the library as `thermo.estimators` +
`thermo.nulls`; it needs to become the front door.

### Slot 5 — there is no interactive QRE tool on the web at all

GTE is the only hosted game-theory solver of any standing. Its primary URL returns 404 from a
2013 Jetty; its surviving mirror is an Angular SPA on an auto-generated free-tier Firebase
subdomain. **It has no QRE, no λ slider, no branch plot.**

A single page with a λ slider, a live branch, and an α/ℛ/σ_EP readout would be the only one
of its kind in the world. SAGE Labs' `/lab` route is already most of the way there and does
not say so anywhere.

**Change:** say so. And add the missing affordances (§3).

---

## 3. The eight friction points, and what each becomes

From the landscape audit's "what does a researcher hand-write that they shouldn't":

| # | Friction today | Change |
|---|---|---|
| 1 | Payoff tensors built by hand from positional arrays; strategy labels re-derived for every plot | `Game.from_frame()`, **named players and actions that survive to the figure axes**, schema validation, a game catalogue |
| 2 | λ standard errors hand-rolled, incomparably, in every paper | `fit().summary()` — Slot 2 |
| 3 | Comparative statics by finite-differencing a re-solved equilibrium — slow, noisy, **silently wrong across a turning point** | `chi_equilibrium` is exact and already gated to 1.3e-8 against finite differences. **This is the pyblp-elasticities-shaped hole and it is already filled; nobody has been told** |
| 4 | Comparing Nash / QRE / level-k / QLk means three libraries, three game objects, two converters | `compare(game, concepts=[...]) -> DataFrame`, one object |
| 5 | Install fails on the referee's machine | Slot 1 |
| 6 | `logit_solve_branch` returns a bare list; every user writes the same unpacking loop and picks a different λ grid, so no two papers' branch figures are comparable | `Branch` as a tidy object with turning points flagged, a serialisation format, and `plot_branch()` |
| 7 | **No `plot_branch()` exists anywhere.** Every simplex plot, bifurcation diagram and λ-sweep panel is hand-rolled matplotlib | `strataq.viz` with one palette, vector output, publication defaults. *arviz became mandatory in Bayesian workflow largely because `plot_trace` existed* |
| 8 | oTree/z-Tree output → estimator is entirely manual, and destroys the panel structure | tidy-frame ingestion + an oTree adapter |

---

## 4. The positioning sentence

> **pygambit computes equilibria. `strataq` tells you what kind of system you are looking at.**

Corollaries that follow from it, and should appear in the README in this order:

1. **We do not compete on correctness of the QRE correspondence.** pygambit is the oracle and
   we validate against it to 1e-8 (`gambit_agreement.json`). Say this loudly; it converts the
   incumbent from a competitor into a certificate.
2. **We compete on everything after the equilibrium** — the derivative (χ), the geometry (α),
   the dynamics (σ_EP), the inference (CIs), the figure, and the verdict.
3. **We compete on install.** Every platform, no compiler.

---

## 5. Who adopts, and what each needs to see first

| Community | The hook | The one thing they need before they will use it |
|---|---|---|
| **Experimental / behavioural game theory** | `fit().summary()` — the missing manual, implemented | Agreement with `pygambit.qre.logit_estimate` on the same data, plus a CI method they can cite |
| **Stochastic thermodynamics** | the first agent-aware entropy-production estimator | Machine-precision agreement on analytically solvable benchmarks, and a demonstration that the estimator is a **valid lower bound** under coarse-graining — not just a number |
| **Econophysics** | a falsifiable market-temperature measure | A worked public-data example, and a discriminant-validity test showing σ_EP is **not** realised volatility or autocorrelation in disguise |
| **Multi-agent RL** | α and σ_EP as training-time diagnostics | JAX-native, `jit`/`vmap`-able inside a training loop at negligible cost, plus evidence that α predicts self-play non-convergence. **They will not adopt a post-hoc analysis tool** |
| **Traffic / congestion** | logit SUE *is* Fisk *is* the potential case | A TNTP/AequilibraE interface and validation on Sioux Falls / Anaheim / Chicago-Sketch against published link flows. **We already have Sioux Falls at ℛ = 5.6e-17 and have never told this community** |
| **Electricity markets** | **irreversibility as a market-power detector on bid data, with no cost estimation** | Validation on a known-manipulation episode (California 2000–01), a bounded false-positive story, and interpretability sufficient for a regulatory filing. *Most likely to pay.* |

**The cross-cutting requirement, from the audit:** every one of these communities must be
able to hand us *their* data in *their* format and get a number with an uncertainty band and
a null-model comparison, without learning game-theory notation. That is `diagnose()` and
`from_trajectories()`. Everything else is second.

---

## 6. What we should stop claiming

- **Stop implying we compete with Gambit on solving.** We do not, we should not, and saying
  we validate against it is worth more than any feature-matrix row.
- **Stop leading with "QRE library."** The category is crowded at the top by an incumbent
  with a 30-year head start and empty everywhere else. Lead with the diagnosis.
- **Stop shipping the calibration results as museum pieces.** Sioux Falls at 5.6e-17 is a
  *credential* aimed at the traffic community, and Dominick's at ℛ = 0.0011 is a *credential*
  aimed at IO economists. Both are currently displayed as project history on a findings page.
