# Paper programme v2 — what gets written, in what order, to what standard

**Supersedes** the four-paper sketch in `THERMOQRE_PROGRAMME_v3.md` §6 and ADR-0013's
p1+p3 split. **Reads with** `research/DIRECTION_v4.md`.

---

## 1. The decision

Two papers, not four, and not the current two.

| | **P-A · The Irreversibility Plane** | **P-B · strataq** |
|---|---|---|
| Kind | flagship research paper | software / tools paper |
| Claim | response asymmetry and dissipation are independent coordinates | these functions compute these things, and here is the calibration |
| Venue | MDPI *Entropy* (Non-equilibrium Phenomena) or *Symmetry* (Physics); arXiv `physics.soc-ph` first | JOSS, or *SoftwareX* / *Journal of Statistical Software* |
| Blocked on | ~~R10′ (the two kill-shots)~~ — both run (F-0022): survives K2 narrowly (41.5% vs a 50% bar), passes K3 at |Δρ_S| = 0.00000 | nothing — writable now |
| Location | `papers/p2_plane/` (drafted, compiles, 17 pp) | `papers/p1_instruments/`, cut down |
| Status of p3 | **absorbed and retired** | — |

### Why two and not three

The current `p1_instruments` and `p3_noneq` share the phase-map and decoupling results
nearly verbatim; a referee reading both will see it. More importantly, neither carries a
claim strong enough to be a flagship: p1's headline theorem is three lines of resolvent
algebra and says so twice, and p3's longest section (28% of the paper) is the project's own
engineering history. Merging their scientific content under one claim, and moving the
software content into a paper whose *job* is software, fixes both problems at once.

### What happens to the existing drafts

- **`p3_noneq` is retired.** §§1–4 and §6 become P-A §§3–5; §5 (`protocols.tex`, 1 313
  words of estimator narrative) is cut to one paragraph on the F-0012 driving-cost
  inversion, which is a genuine world-fact and belongs in a future quench paper, not here.
- **`p1_instruments` becomes P-B.** Cut the findings section (it duplicates P-A), keep the
  calibration table, the reproducibility section and the honest-positioning section. A
  software paper's calibration table is its argument; that table is already the best thing
  in the repository's writing.

---

## 2. Four things that must be fixed before anything is sent anywhere

These are not stylistic. Both current abstracts contradict the project's own record.

1. **`p1` abstract says ℛ is "a payoff-free, λ-free *test*"** without the qualifier that
   only the *zero* test is λ-free. F-0002 corrected exactly this and the body already gets
   it right. **Fix the abstract.**
2. **`p3` abstract says the five-minute series gets "an equally instructive *certified*
   at-null verdict."** F-0008's certified-null claim was **retracted**; F-0009 downgrades it
   to "at-null in the matched class." The body gets it right. **Fix the abstract.**
3. **`p1`'s provenance appendix claims the findings log covers F-0001–F-0007.** It is stale
   by fourteen findings.
4. **`p3`'s discussion still lists as open** the coincidence between the decoupling onset and
   the supercritical wedge. F-0010's addendum debunked it (0/180 games supercritical).

And one omission worth more than all four: **Dominick's ℛ = 0.0011 (F-0011) appears in the
claims ledger and the app but in neither paper**, despite being arguably the most persuasive
number in the repository. It is in P-A §5.4.

---

## 3. P-A: what is drafted

`papers/p2_plane/` compiles to 13 pages with five figures. Written to the construction
template extracted from the exemplar papers.

**Structure** (MDPI skeleton, evidence-weighted): abstract (340 w, with the scope fence in
the middle, not at the end) → §1 Introduction (7-beat arc, dictionary paragraph, C1–C5,
H1–H4 registered with tests and levels) → §2 Background + **Gap Analysis + capability
matrix** → §3 The Two Coordinates (short; the formalism section is the shortest substantive
section by design) → §4 Experimental Setup (families, solvers, metrics, statistics, nulls,
pre-registration) → §5 Results, the largest section, ending in a **hypothesis scoreboard
with two "Not supported" rows** → §6 Discussion, with *"what ℛ is not"* and a Limitations
subsection at ~40% of the discussion → §7 Conclusions → back matter in MDPI's fixed order →
Appendix A, claim-to-artifact map.

**Figure program** — five figures, all vector, all regenerated from committed artifacts by
`papers/p2_plane/make_figures.py`, all sharing one palette with `strataq.viz` and the app:

| Fig | What it does | Source artifacts |
|---|---|---|
| 1 | schematic: the two computational routes from a game to a coordinate, sharing only a zero | — |
| 2 | **the plane** — the synthetic α-family running up the diagonal (what a scalar theory predicts) with the real systems *off* it, and the empty quadrant named | `decoupling_mechanism`, `estimator_alpha_sweep`, `pricing_passthrough_R`, `electricity_irreversibility_dam`, `blotto_readings`, `sioux_falls_calibration` |
| 3 | **the money figure** — the conditional collapse, with the refuted repair hypothesis plotted alongside it and the marginal value as a dashed reference; the mechanism panel beside it | `chain_comovement`, `decoupling_mechanism` |
| 4 | criticality escape and the descending frontier | `phase_map_surface`, `frontier_lambda_c` |
| 5 | observability — trajectory estimators against the exact meter, and tightness | `estimator_alpha_sweep` |
| 6 | **the honesty exhibit** — the collapse at every m with the terminal value moving to zero, beside the fixed-scale arm against the constant-RMS control that shows the m-trend is temperature | `plane_robustness` |

Figure 3 is the money figure for the same reason the exemplar's was: it plots the whole
field against an explicit reference *and it is honest by construction* — the same axes that
carry the claim carry the refuted repair. A referee cannot accuse the paper of hiding the
negative result because it is the other series in the headline figure.

**Devices carried over from the exemplars, deliberately:**
- The scope fence sits in the abstract, not only in the discussion.
- The headline number never appears without its counter-number (0.993 marginal always
  travels with the conditional value — now +0.01 [−0.14, +0.16] at m = 6, with −0.355
  retained only inside its (m, λ) scope, per F-0022).
- Two comparison tables doing opposite jobs: a **capability matrix** early (earns novelty)
  and no performance leaderboard at all (there is nothing to beat, and claiming one invites
  a fight we would lose).
- A hypothesis scoreboard instead of a boxed theorem, with H2a and H3b recorded as **Not
  supported**.
- Limitations as an ordinal march, each ending in a remedy actually registered or a named
  next step, including one confessed gap: **no system has both coordinates measured yet.**

---

## 4. P-B: what to cut it to

A software paper's job is: what does it compute, is it right, can I install it, can I cite
it. Roughly 3 000 words.

1. Statement of need — the gap analysis from `docs/product/COMPETITIVE_POSITION.md`: no
   published implementation of the Candogan decomposition anywhere; no equilibrium
   derivative in any solver; irreversibility estimators that cannot see agents.
2. Design — the two engines, the plugin contract, the tangent-space discipline, float64,
   and `ConjugateFieldSpec.NONE` as a real refusal.
3. **The calibration table** — ten rows, machine-precision agreements, each mapped to a
   committed artifact. This is the paper's argument; do not shorten it.
4. Validation against the reference solver to 1e-8, and χ against finite differences to
   1.3e-8.
5. Reproducibility — the pre-registration discipline, `make reproduce`, the release-integrity
   story (F-0018: 23 gates green while every solver call in the shipped wheel was broken, and
   the four tests added because of it). **That story is a credential, not an embarrassment**,
   and software-paper referees respond to it.
6. Limitations — dense generator only for small games; `golden/` currently empty; the public
   surface is broader than `__all__` documents.

---

## 5. Order of operations

1. **Fix the four contradictions in §2** — one afternoon, and nothing should be sent before
   it.
2. **Write P-B and submit it.** It makes no scientific claim beyond calibration, so it is
   not blocked on R10′ and it establishes the citable software artifact P-A will reference.
3. ~~**Run R10′**~~ — done (F-0022): the claim survives, narrowed to decorrelation rather
   than sign reversal, and the m-sweep's apparent finite-size decay turned out to be a
   temperature confound of the generator. The independent prior-art re-audit is still open.
   The next kill-shot is N > 2, which is untested.
4. **Submit P-A.** arXiv first, journal second.
5. **Retire p3**, and record in an ADR that the quench line (F-0012 → F-0021) is a successor
   programme with its own paper, not a section of this one.

---

## 6. The standard, condensed to a checklist

Applied to P-A; reusable for anything after it.

- [ ] The claim is one sentence that already contains its own limitation.
- [ ] The strongest simple alternative reading is stated and its number sits next to ours.
- [ ] The scope fence is in the abstract.
- [ ] The claim is restated in five places, always with its counter-number.
- [ ] A dictionary paragraph maps every physics term to its game-theoretic referent.
- [ ] Contributions labelled C1–C5, the last one reproducibility artifacts.
- [ ] Hypotheses stated with test, level and n *before* the results, and at least one
      recorded as **Not supported**.
- [ ] §1 and §2 contain no figures; the capability matrix is the only early exhibit.
- [ ] Formalism ≤ ~10% of words; derivations in an appendix.
- [ ] Results ≈ 25% of words, ending in a scoreboard.
- [ ] A sensitivity sweep on every free parameter, naming our own most fragile knob.
- [ ] Every figure vector, one palette, dashed black reserved for bounds, shared legend
      below the panel pair, captions of one sentence that never restate a result number.
- [ ] Money figure shows the win and the loss on the same axes.
- [ ] Limitations ≈ 40% of the discussion, ordinal, each ending in a remedy or a named step.
- [ ] One confessed gap stated as *"a real limitation rather than evidence of generality."*
- [ ] A subsection killing the flattering misreading of our own metric.
- [ ] Back matter in MDPI's fixed order, AI-tool disclosure in acknowledgments, all access
      dates identical.
- [ ] 40–60 references; every non-DOI source access-dated.
- [ ] Every number traceable to a committed artifact in the provenance appendix.
