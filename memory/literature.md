# Literature — annotated, with what is settled vs contested vs open

Prior-art checks live here. Novelty claims never survive without a sweep recorded on this page. The full nearest-live-work report: [`literature-nearest-live-work.md`](literature-nearest-live-work.md) (2026-08-08).

## Nearest live work

**arXiv:2405.07224** — Legacci, Mertikopoulos & Pradelski, "A Geometric Decomposition of Finite Games: Convergence vs. Recurrence under Exponential Weights" (ICML 2024). Continuous-time replicator (exponential weights) under Shahshahani geometry: harmonic ⟺ incompressible; harmonic games are volume-preserving, admit constants of motion, and exhibit Poincaré recurrence. **Explicitly open**: discrete-time schemes and mixed potential/harmonic games. **Verdict (theory-verifier, 2026-08-08): orthogonal, not competing.** They classify long-run dynamics; we build the equilibrium-response layer (χ^eq, ℛ) and the thermodynamic observable layer (J*, EPR, TUR) for discrete-time logit QRE, with empirical contact. Recommendation: cite prominently as SOTA on the decomposition-dynamics layer, position ThermoQRE as filling their open discrete-time question at the observable level; no collaboration needed. Draft related-work paragraph is in the full report §4. Also flags Lemma C.2 / Example C.2 (full-vs-effective potentiality) — the precision fix baked into K4 and `finite/CLAUDE.md`.

## Settled (foundations we implement, cite, never claim)

- **QRE**: McKelvey–Palfrey *GEB* 1995 (definition, existence); *ExpEcon* 1998 (AQRE); Turocy *GEB* 2005 (logit homotopy — Gambit's method); Goeree–Holt–Palfrey 2005/2016 (regular QRE axioms).
- **MaxEnt bridge**: Gibbs variational principle; WDZ theorem; Fudenberg–Iijima–Strzalecki 2015 (perturbed utility); Matějka–McKay 2015 + Fosgerau–Melo–de Palma–Shum 2020 (rational inattention ⟹ logit, λ = information price).
- **Potential games & decomposition**: Monderer–Shapley 1996; Blume 1993 (logit/Glauber, Gibbs stationarity); Candogan et al. *MOR* 2011 (flow decomposition — source of α); Sandholm 2001/2010 (externality symmetry, population games); Balduzzi et al. ICML 2018 (Jacobian splitting).
- **Congestion**: Rosenthal 1973 (exact potential); Beckmann 1956; Fisk 1980 (SUE = entropy-regularised Wardrop — the analytically known potential with real data). TNTP networks with best-known equilibria.
- **Stochastic thermodynamics**: Jarzynski, Crooks, TUR, Hatano–Sasa — textbook; Otsubo et al. *Comms Phys* 2022 (EPR estimation from trajectories, arXiv:2010.03852).
- **Solvers**: Sokota et al. ICLR 2023 (magnetic mirror descent, last-iterate linear convergence to QRE).
- **Demand/IO**: BLP 1995; Conlon–Gortmaker 2020 (pyblp); HHK *AER* 2008 (the non-falsifiability threat); Duarte–Magnolfi–Sølvsten–Sullivan *QE* 2024 (conduct testing, pyRVtest, weak-instrument diagnostic); Calvano et al. *AER* 2020 (algorithmic collusion).

## Contested / cautionary

- **QRE's empirical content**: HHK say none without restrictions. Our three-layer defence: regular-QRE axioms; externally identified payoffs; the λ-free reciprocity restriction (lead with the third). Any claim ignoring this gets killed in review.
- **Pure QRE loses horse-races**: Wright–Leyton-Brown *GEB* 2017 — QLk beats pure QRE across ten datasets. Logit QRE is our *baseline*, never a presumptive winner; ablations always include level-k/QLk.
- **Hopf under logit in harmonic games is real**: Hommes–Ochea 2012 (RPS limit cycles). Supports N3's contrapositive; also means cycle-hunting at α > 0 has precedent.
- **QRSE** (Scharfenaker–Foley 2017): the nearest econophysics analogue; aggregate-outcome MaxEnt with self-acknowledged non-identification and equilibrium-only scope. We are game-theoretic, payoff-identified, and add the current/EPR structure QRSE cannot represent. Position against it, don't ignore it.
- **Dominick's caveats**: single chain, zone pricing, 1989–97; rival-firm interpretation is a modelling assumption (the programme's weakest empirical link — why congestion/Blotto/ERCOT carry the structural weight). Gandhi–Lu–Shi 2023: zero-share handling can double elasticities. Xu & Wang arXiv:1107.6043: the closest empirical precedent (EPR identifies an Edgeworth cycle in experimental data) — read before any empirical EPR claim.

## Open (nobody has this; our territory if we execute)

- The equilibrium-response layer as an observable of potentiality (N1/N2) — no prior statement found (sweep 2026-08-08).
- Discrete-time logit QRE behaviour across the α axis with thermodynamic observables — 2405.07224 leaves discrete time open.
- λ-free harmonic-content estimation from pass-through asymmetry on real data.
- Cross-domain instrument comparison (same meters: congestion 0, Blotto high, Bertrand between).

## Watch list

- arXiv:2608.01967 (second-order potentials / MS-potential decomposition) — alternative decomposition; revisit if games with higher-order structure matter.
- Follow-ups citing 2405.07224 — re-sweep before submitting p2_reciprocity.
