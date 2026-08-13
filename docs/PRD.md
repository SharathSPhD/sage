# SAGE — Product Requirements, living

**Status:** authoritative. Supersedes `docs/product/PRODUCT_v1.md` where they disagree.
**Version:** 0.1 · 2026-08-13 · **this document is expected to change; see §9.**

---

## 0. The correction this document exists to make

Two framings have to die, both of them mine.

**"The project is an instrumentation programme."** It is not. Instruments are *one
component*. `research/THERMOQRE_PROGRAMME_v3.md` §0.1 said "exploratory instrumentation
building" and the machine executed it until it was measuring its own thermometer. The
instruments are the means. The project is about **what strategic systems do, and what to do
about it.**

**"Diagnosis, not prescription."** `PRODUCT_v1.md` §2.1 wrote that down as a virtue. It is
the ivory tower in one phrase. Telling somebody their market is a whirlpool and then
declining to help them act is not rigour; it is abdication. A refusal is honest only when it
is *specific* — this quantity is unidentified by your data, here is what would identify it —
never as a general posture.

**What replaces them:** the product exists to **produce solutions**. A strategic situation
goes in; a decision, with its uncertainty and its reasoning, comes out. The measurement is
how we earn the right to make the recommendation, not the deliverable.

---

## 1. What this is

A system for **solving strategic situations** — pricing, bidding, routing, allocation,
coordination — where the other side is also choosing, and nobody is perfectly rational.

Three layers, in dependency order. Each is a component, none is the product.

| Layer | What it is | Not |
|---|---|---|
| **Engine** (`strataq`) | solvers, decomposition, response, dynamics, estimation | not the product |
| **Instruments** | α, ℛ, σ_EP, λ — how we know the model fits reality | not the product |
| **Solutions** | the recommended action, its alternatives, its risk, its story | **this is the product** |

---

## 2. Who it is for, and what they get

### 2.1 The practitioner
> "Two rivals and I set prices weekly. What do I do on Monday, and what happens if I'm wrong?"

Delivered: a recommended action; the distribution of what rivals do; expected outcome with
an interval; the two or three alternatives and what they cost; what has to be true for the
recommendation to hold; and what would change it.

### 2.2 The researcher
> "I ran an experiment. Fit the model, tell me whether it's the right model, give me the figure."

Delivered: `fit()` with real inference, model comparison against Nash / level-k / uniform,
publication figures, and a reproducible artifact.

**Neither is served by a page that reports α = 0.69.** That number is evidence for a
recommendation, not a recommendation.

---

## 3. The app — a solution studio, not a readout

The app is where solutions are **produced and played with**, live.

### 3.1 Non-negotiables
- **Every surface ends in a decision or an insight the visitor can use.** No panel exists
  only to display an instrument reading.
- **Live and interactive.** Drag a payoff, move a slider, watch the solution move. Sub-100ms
  for exploration; the authoritative solver for the committed answer.
- **Plain language first.** No visitor should need "harmonic fraction" to get value. The
  technical name is available on demand, never in the way. **If a sentence would not survive
  being read aloud to a pricing manager, it is cut.**
- **Multiple situations, not one.** Pricing duopoly, procurement/auction, congestion and
  routing, resource allocation (Blotto), coordination and standards, entry and capacity.
  Each is a worked, playable scenario with real numbers.

### 3.2 Surfaces
1. **Solve** — pick a situation from a gallery or build one. Get a recommended action,
   rival distribution, payoff surface, and the sensitivity that matters. This is the home
   page, not a sub-page.
2. **Play** — the live demos. Adjust anything, see the solution move, race it against
   baselines (best response to last move, tit-for-tat, cost-plus, always-Nash) and watch who
   wins over repeated rounds. **This is what makes it obvious rather than argued.**
3. **Bring your data** — upload; fit; get a solution calibrated to your situation, with the
   honest statement of what the data can and cannot support.
4. **Bench** (researcher) — solve, fit, compare concepts, export figures and a citable
   artifact.
5. **Learn** — the existing explainers, kept, reframed so each one ends in "so here is what
   you'd do differently".

### 3.3 Verbiage
Current copy fails both audiences. Rules: lead with the decision; name the situation the way
its practitioners name it; put every Greek letter behind a hover; never use "instrument",
"meter", "reading", or "certified" in primary copy; and no sentence about the programme's own
epistemics on a solution surface.

---

## 4. Science

The instruments serve claims about the world. Standing agenda in `research/DIRECTION_v4.md`.

- **Live claim.** ℛ and σ_EP are independent coordinates. Ceiling result robust across
  m ≤ 6 and N ≤ 4. Sign reversal survives m once λ̄ is matched (F-0024). **Low-α co-movement
  is an N=2 fact** — carry that scope everywhere.
- **Open, in priority order:** a real system in quadrant III; K4 prior-art re-audit;
  N>2 with a design whose precondition can actually adjudicate; the paired-control re-run
  (R11 O-3/O-6); gate `science.plane` and re-review it.
- **Rule earned the hard way:** a unit registers criteria **once**. A second criteria file is
  a substitution and must be disclosed with both adjudications (F-0022).

---

## 5. Library and API

- `diagnose()`, `fit()`, `viz` shipped. Next: `solve_situation()` — the solution-shaped entry
  point the app is built on (recommended action + rival distribution + alternatives +
  sensitivity), so the app never assembles a recommendation itself.
- Wheel stays `py3-none-any`; installs with no compiler on every platform.
- API mirrors the library one-for-one.

## 6. Papers

Flagship (`p2_plane`) carries the claim. Software paper (P-B) unwritten. **p1 and p3 still
carry abstracts that contradict the record and are on `main` — fix before anything is
submitted** (`papers/PAPER_PROGRAMME_v2.md` §2).

## 7. What "done" means

1. A stranger solves their own situation and can say what they'd do differently.
2. Someone cites `fit()` for a λ interval they didn't hand-roll.
3. Three real systems located, in two quadrants, each with a CI and a registered null.
4. Flagship and software papers submitted.
5. Every claim on `main` is gated or explicitly recorded as un-gated.

## 8. Explicit non-goals

Not an autonomous pricing system. Not a Gambit competitor on solving — we validate against
it. Not a general MARL framework. Not a dashboard.

---

## 9. How to keep this document alive

- **It changes.** When intent and this document disagree, the document is wrong — update it
  in the same commit as the work.
- Bump the version and add a line to §10 for every substantive change.
- Anything contradicting §0 is a regression and should be reverted or argued in an ADR.
- Reviewed at the close of every unit, alongside `SESSION.md`.

## 10. Revisions

- **0.1 · 2026-08-13** — created. Kills "instrumentation programme" and "diagnosis, not
  prescription". Reframes the app from a diagnostic readout to a solution studio with live
  playable demos. Records the N=2 scope restriction from F-0023/F-0024.
