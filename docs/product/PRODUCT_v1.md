# StratAQ / SAGE Labs — Product v1: two front doors, one engine

**Supersedes** `THERMOQRE_PROGRAMME_v3.md` §10 (Learn / Lab / Analyze).
**Reads with** `research/DIRECTION_v4.md` (what the product delivers) and
`docs/product/COMPETITIVE_POSITION.md` (why anyone switches).

---

## 0. The failure being fixed

The current app is a museum of the research. Nine routes, and eight of them are *about the
project*: a scripted tour, a findings gallery, a pre-computed phase heatmap, two calibration
demos, ten explainers. Exactly one route (`/tools`) accepts anything the visitor owns, and it
accepts it as pasted text into a `<textarea>` and a grid of `<input>` boxes.

The measured consequences, from the audit:

- **No file upload anywhere.** No `<input type="file">` in 3,693 lines of TSX.
- **No export of anything.** No download, no `Blob`, no CSV, no copy-to-clipboard.
- **No "reproduce this in Python."** v3 §10.2 specified it as "the main funnel from app to
  library." It was never built. `/tools` says the words `pip install strataq` in prose.
- **No chart library.** Every visualisation is hand-cut inline SVG. This is not a virtue; it
  is why there are so few of them and why none of them are publication-grade.
- **`game_thermo`** — the one facade call that produces the actual scientific verdict — is
  the **only** toolkit entry point with no `warnings` field and the **only** one not exposed
  over HTTP.

So: a visitor arrives, is told a story about someone else's markets, and leaves. There is no
moment at which the product touches their problem. That is the whole of the "ivory tower"
complaint and it is correct.

---

## 1. TRIZ: the contradiction, and why the current design lost to it

### 1.1 The technical contradiction

- **Improving:** #28 Measurement accuracy — refusals, confidence tiers, calibrated bands,
  uncertainty intervals, null models, "do not publish a classification from this."
- **Worsening:** #33 Ease of operation, #39 Productivity — a practitioner needs a decision
  now, and every honesty affordance is a step between them and it.

Matrix (28→33): **1 Segmentation · 13 The Other Way Round · 17 Another Dimension · 34
Discarding & Recovering.**
Matrix (28→39): **10 Preliminary Action · 34 · 28 Mechanics Substitution · 32 Colour Change.**

### 1.2 The physical contradiction

> The same reading must be **decisive** (one verdict, actionable immediately) and
> **non-decisive** (bounded, caveated, refusable, publishable).

Separation principle that applies: **by system level** — *the whole has one property, the
parts have the opposite.* The verdict is a single word. Every component of it is a bounded,
refusable measurement. Secondary: **separation in time** — the practitioner gets the verdict
in the first second; the audit trail is one click later, not one scroll later.

The current design resolved this contradiction *backwards*: it made every part decisive
(a number on a gauge) and the whole non-decisive (nine routes, no conclusion).

### 1.3 The seven resolutions, and what each becomes in the build

| TRIZ | Reading | Concrete change |
|---|---|---|
| **17 Another Dimension** | Stop projecting a two-dimensional thing onto one axis. | The UI's primary object is **the (ℛ, σ_EP) plane**, not a row of gauges. A gauge shows a number; the plane shows *a position*, which is a decision. This is the same move as the research result — the product and the science become the same shape. |
| **13 The Other Way Round** | Invert refusal into information. | **Never return "cannot compute."** Return the widest true statement: a bound, a quadrant *edge*, a "your data can distinguish I from IV but not II from III, and here is the n that would." R10's amended T1 already discovered this (MEASURED / **BOUNDED** / REFUSED); generalise it to every instrument. |
| **1 Segmentation** | Split the instrument, not the audience. | Three tiers behind one call: **instant** (client-side TS, already exists in `lib/qre.ts`), **fast** (API float64, seconds), **certified** (queued, with nulls and bootstrap). Same object, same numbers, escalating warrant, badge shows which. The visitor never chooses a tier; they choose *how sure they need to be*. |
| **34 Discarding & Recovering** | Caveats are discarded from the headline and recovered on demand. | The verdict line carries the word and the CI. Every warning, tier, null, seed and provenance stanza lives behind one **"Why should I believe this?"** control that is always present and never open by default. |
| **10 Preliminary Action** | Pre-compute what makes the first second fast. | Ship a **precomputed reference cloud** — every gated artifact plotted as faint points in the plane — so the visitor's own point lands *in context* immediately, next to Sioux Falls, Blotto, RPS, Dominick's, CAISO. Their reading is never a naked number. |
| **32 Colour Change** | Make state visible without reading. | Four quadrant colours, used identically in the app, the library's `repr`, the exported figures and the paper. One semantic palette across every surface. |
| **28 Mechanics Substitution** | Replace the manual field with the file. | Kill the paste-a-matrix grid. **CSV/Parquet drop → schema inference → validation report → reading.** The validation report is where the honesty goes, and it is *useful* rather than obstructive because it says what the data can and cannot support. |

---

## 2. The two audiences, stated as jobs

Neither audience wants "a QRE library." Each has a job it currently does badly.

### 2.1 The practitioner

> *"I have a panel of what my competitors did. Is my market a landscape or a whirlpool, and
> what does that change about what I should do on Monday?"*

Today they cannot ask this of anything. The alternatives are: read the pass-through
literature and hand-code an asymmetry regression, or read the irreversibility literature and
run a scalar time-series test that has no concept of agents. Neither answers the question
they actually have, which is *what kind of thing am I in*.

What makes it a must-have: **the quadrant changes the recommendation, and the recommendation
is legible without game theory.**

| Quadrant | What it means for Monday |
|---|---|
| **I. Landscape** | Optimise. Comparative statics hold, pass-through is symmetric, a static competitor model is adequate, there is nothing to time. |
| **II. Driven landscape** | Something exogenous is cycling you — demand, schedules, cost. **Time your moves; do not re-engineer your strategy.** Structure is fine. |
| **III. Stalled whirlpool** | A rival structurally leads. **Asymmetric pass-through is the exploitable object.** There is no cycle to time, so timing effort is wasted. |
| **IV. Whirlpool** | Both. Optimising against a static rival model is at its worst here. Model the rival's *distribution*, and expect cycles. |

Note what this is *not*: it is not a pricing optimiser and it does not claim to beat a tuned
demand model at setting tomorrow's price. v3 §2 was right to disclaim that and
`docs/theory/09-the-one-price-objection.md` is right to state the PI's own objection. The
product's honest pitch is **diagnosis, not prescription** — and diagnosis is a job nobody
else is doing at all.

### 2.2 The researcher

> *"I ran an experiment / I have a panel. Fit QRE to it properly, tell me whether QRE is even
> the right model here, and give me the figure."*

The definitive statement of this gap is that Gambit's own maintainer had to publish
**Bland & Turocy (2025), "Quantal response equilibrium as a structural model for estimation:
the missing manual," *GEB***. When the maintainer of the only QRE solver writes a *manual*
for the estimation workflow, the workflow is not in the software. It is in prose, and every
lab re-implements steps 4–8 of it from scratch.

What makes it a must-have, in order of pain:

1. **`fit(game, data).summary()`** — λ with a confidence interval, an LR test against Nash
   and against uniform, and a fit diagnostic. `pygambit.qre.logit_estimate` returns `.lam`
   and `.log_like` and nothing else; every published CI on λ is hand-rolled and no two labs
   do it the same way.
2. **Panel structure survives estimation.** Gambit's input is a `MixedStrategyProfile` of raw
   counts, so subject identity, round and treatment are destroyed *before* the likelihood is
   written. Accept the tidy frame. Keep the structure. Offer hierarchical λ.
3. **`plot_branch()` and `plot_plane()`.** There is no branch plotting function anywhere in
   the ecosystem. arviz became mandatory in Bayesian workflow largely because `plot_trace`
   existed.
4. **Install that works.** pygambit has shipped **zero** Linux and **zero** macOS wheels
   across every 16.x release — twelve Windows wheels and an sdist. Every Mac, every Linux
   box and every Colab notebook compiles C++17 from source. A pure-Python/JAX wheel is an
   unearned win on every platform the incumbent does not serve.
5. **α, ℛ, σ_EP as diagnostics.** No PyPI package implements the Candogan decomposition —
   fifteen years after the founding paper. The slot is empty.

---

## 3. One engine, two front doors

```
                 ┌──────────────────────────┐   ┌──────────────────────────┐
   FRONT DOORS   │  /diagnose               │   │  /bench                  │
                 │  "what kind of system    │   │  "fit it, test it,       │
                 │   am I in?"              │   │   plot it, cite it"      │
                 │  drop a CSV → a quadrant │   │  notebook-shaped,        │
                 │  → what that changes     │   │  export-first            │
                 └───────────┬──────────────┘   └────────────┬─────────────┘
                             └────────────┬──────────────────┘
                 ┌────────────────────────▼─────────────────────────┐
   ONE FACADE    │  strataq.diagnose(...)  ·  strataq.fit(...)      │
                 │  same objects, same numbers, same provenance     │
                 └────────────────────────┬─────────────────────────┘
                 ┌────────────────────────▼─────────────────────────┐
   ONE ENGINE    │  strataq — solvers · decompose · response ·      │
                 │  dynamics · thermo · estimate · domains          │
                 └──────────────────────────────────────────────────┘
```

The two doors are **not** two apps and **not** a simple/advanced toggle. They differ in the
*question asked*, not the depth of answer. A practitioner who clicks "why should I believe
this?" arrives at exactly the researcher's evidence, in the researcher's units.

### 3.1 `/diagnose` — the practitioner door

Five states, and the visitor must reach state 3 within sixty seconds of arriving.

**1 — Drop.** A dropzone that accepts CSV/Parquet, plus three one-click sample datasets
(Dominick's, CAISO, a synthetic whirlpool) so a visitor with no data still reaches a verdict.
*This replaces the textarea. TRIZ 28.*

**2 — Recognise.** Column-role inference (entity, time, action/price, optional cost,
optional group) with an editable mapping, and a **validation report** stating what this data
can and cannot identify: panel balance, action-grid density, cross-sectional variation, the
n needed for each axis. *Refusals become bounds here. TRIZ 13.*

**3 — The plane.** One figure, dominant, above the fold. Their point, with error bars, in the
(ℛ, σ_EP) plane, on top of the faint precomputed reference cloud of every calibrated system
the project has. Quadrant shaded and named. One sentence: **"Landscape (ℛ = 0.0011
[0.00005, 0.0050], σ_EP at null). Read at λ̂ = 1.4."** *TRIZ 17, 10, 32.*

**4 — What that changes.** Three to five plain-language consequences from the quadrant table
in §2.1, each with the specific number that drove it, and each with a named check the visitor
can run themselves to see if we are wrong.

**5 — Take it with you.** Three buttons, always present:
- **Reproduce in Python** — a runnable `strataq` snippet with their column mapping baked in
  and the same seed. *This is the funnel v3 specified and the build skipped.*
- **Export** — figure as vector PDF + PNG, readings as CSV, full provenance as JSON.
- **Why should I believe this?** — the drawer: every warning, every tier badge, the null
  model, the CI method, the seed, the library version, the calibration anchors, and the link
  to the finding that established each band. *TRIZ 34.*

### 3.2 `/bench` — the researcher door

Notebook-shaped, four panes, export-first.

- **Game.** Build from a tensor, a named catalogue entry, a demand oracle over a price grid,
  or an uploaded payoff CSV — with **named players and named actions that survive all the way
  to the figure axes.** (Today `Game.from_arrays` takes positional arrays and the labels are
  re-derived for every plot, by hand, in every project.)
- **Solve.** λ slider, branch with turning points marked, spectrum of SB against the unit
  circle, α decomposition bar, ℛ, σ_EP. Every panel exports.
- **Fit.** Upload experimental data → `fit().summary()`: λ̂ with CI, LR against Nash and
  uniform, per-group λ, agreement across the four estimators with the disagreement flagged
  rather than averaged away.
- **Compare.** One table, one game, several concepts: Nash, logit QRE, level-k, QLk. This is
  friction point #4 in the landscape audit and nothing in the ecosystem does it, because each
  concept lives in a different library with a different game object.

Every pane carries the same **Reproduce in Python** button. The app is a *specimen generator*
for the library, not a replacement for it.

### 3.3 What happens to the existing routes

| Route | Disposition |
|---|---|
| `/lab` | Becomes `/bench` "Solve". Keep the payoff editing and the λ slider; add export and the snippet. |
| `/learn`, `/story` | Keep. They are good and they are the single-source of `docs/theory`. Move `/story` to end on **"now try it on your data"** with a link to `/diagnose`, not on a summary. |
| `/phase` | Absorbed into `/bench` as a live layer, not a static SVG. It is currently the clearest example of the museum problem: a pre-computed picture with no click-through. |
| `/findings` | Keep, demote from the nav to a footer link. It is documentation of the programme, not a product surface. Add the seven missing findings. |
| `/network`, `/blotto` | Keep — these are the two routes where a visitor genuinely manipulates a real system and watches the meters move. They become **calibration exhibits** inside `/bench`, reachable from `/diagnose`'s "why should I believe this?" as *"here is the instrument reading zero on a system where zero is provably right."* |
| `/markets` | Absorbed into the reference cloud as one point plus its detail view. |
| `/tools` | **Deleted.** Its two functions are `/diagnose` done badly. |

---

## 4. The library surface this requires

Three additions. Nothing else in `strataq` changes.

```python
# 1. The one call. Three input shapes, one return type.
strataq.diagnose(payoffs=..., lam=...)  # a game you specify
strataq.diagnose(panel=df, roles={...})  # observed behaviour
strataq.diagnose(chi=..., trajectory=...)  # readings you already have
# -> Diagnosis(quadrant, r, r_ci, epr, epr_ci, epr_null, lam, alpha,
#              tier, warnings, refusals, provenance, reference_cloud)
```

`Diagnosis.__repr__` prints the verdict line. `.explain()` prints the drawer.
`.snippet()` returns the runnable reproduction. `.plot()` returns the plane figure.
**Every field that can be a bound rather than a refusal is a bound.**

```python
# 2. The estimation workflow the field is missing (Bland & Turocy 2025).
fit = strataq.fit(game, data, by="subject")  # tidy frame in, panel structure kept
fit.summary()  # lambda_hat, CI, LR vs Nash, LR vs uniform, n, method, warnings
fit.plot()  # branch with lambda_hat marked and the data overlaid
```

```python
# 3. Plots that are publication output, not debug output.
from strataq.viz import plot_plane, plot_branch, plot_decomposition, PALETTE
```

One shared palette, vector output, the same colours as the app and the paper.

### 4.1 The README first block

The landscape audit is unambiguous about what the first five minutes must do: ≤ 7 lines, no
data file, and output a human reads as an **answer**, not an array. Current best-in-class is
`axl.Tournament(players, seed=1).play().ranked_names`. Target:

```python
>>> import strataq
>>> rps = strataq.games.rock_paper_scissors()
>>> strataq.diagnose(rps, lam=1.5)
Diagnosis: WHIRLPOOL  (quadrant IV)
  response asymmetry  R = 0.87        [0.83, 0.91]
  dissipation       EPR = 0.42 nats/step  (null: 0.00)
  harmonic fraction   a = 1.00
  read at lambda = 1.5 · tier: certified · 0 refusals
  -> circulation and asymmetric response: model the rival's distribution, expect cycles
```

Compare with what it displaces: `pygambit` needs numpy plus two hand-transposed arrays before
anything happens and returns a nested list of floats; `sgamesolver` needs a four-call
ceremony. One call, one verdict.

---

## 5. Build order

Each step is independently shippable and each ends with something a stranger can use.

1. **`viz` + palette.** Everything downstream needs figures. Also unblocks the paper.
2. **`diagnose()` + `Diagnosis`,** with `explain()`, `snippet()`, `plot()`. Add the missing
   `warnings` field to `game_thermo` on the way past.
3. **README first block, PyPI wheels for every platform.** Cheap, and it is the whole
   adoption story against the incumbent.
4. **`/diagnose` route** — dropzone, inference, validation report, plane, consequences, the
   three buttons. Kill `/tools`.
5. **`fit()` + `summary()`,** the Bland–Turocy workflow. Highest-value single deliverable for
   the researcher audience.
6. **`/bench`,** absorbing `/lab` and `/phase`.
7. **oTree / experimental-data adapter,** which is where the researcher audience already is.

---

## 6. How this gets judged

Not by gates. By these:

- A stranger reaches a quadrant verdict on their own data in under sixty seconds.
- A stranger who reaches a verdict can say, unprompted, what it changes about what they do.
- Someone cites `strataq` for a λ confidence interval they did not hand-roll.
- A figure produced by `strataq.viz` appears in a paper that is not ours.
- The reference cloud gains a point somebody else put there.

None of those are achievable by the app as it stands, and all five are achievable from the
engine as it stands. The science is not the bottleneck. The last hundred metres is.
