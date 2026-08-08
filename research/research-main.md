# A ThermoQRE Research Programme for Discrete-Price-Grid Oligopoly: QRE, MaxEnt, and Non-Equilibrium Thermodynamics — Literature Map, Theory, Data, and Solvers

## TL;DR
- The exact, defensible core of the programme is the chain of identities **logit-QRE = entropy-regularised Nash = Gibbs measure over the potential (in potential games)**, with λ = inverse temperature β; the genuinely novel, publishable result is that **demand/price response is a susceptibility equal to a choice-probability fluctuation (a static fluctuation–dissipation identity)** — a framing already proved for the consumer-side Slutsky matrix by Garnier-Brun, Bouchaud & Benzaquen (*Journal of Physics: Complexity* 4(1):015004, 2023) and which extends cleanly to the firm-side logit-QRE, making a "dispersion–response estimator of λ" theoretically well-founded.
- The Hugging Face dataset the user calls "dreamprice" exists as **`qbz506/dreamprice-dominicks-cso`** (Dominick's canned-soup scanner panel, 500k rows, 529 columns, CC-BY-NC-4.0), a processed slice of the canonical Dominick's Finer Foods database; both are documented below with exact schema and the "qty bundle" gotcha, and mapped to model objects.
- The Haile–Hortaçsu–Kosenok non-falsifiability theorem is fatal to QRE *in the abstract* but is defused when payoffs are pinned down by an independently estimated demand system; the programme must therefore be a two-stage "estimate demand (BLP), then test λ as a conduct parameter" design, which reframes λ-estimation as the Duarte–Magnolfi–Sølvsten–Sullivan conduct-testing problem.

---

## Key Findings

1. **The MaxEnt/QRE equivalence is an exact mathematical identity, not an analogy.** The logit choice rule is the argmax of expected payoff plus (1/λ)·Shannon entropy; the log-partition function (log-sum-exp / McFadden inclusive value) is the cumulant generating function whose gradient gives choice probabilities (Williams–Daly–Zachary theorem) and whose Hessian gives the choice-probability covariance. This is exactly the discrete-choice static fluctuation–dissipation relation.

2. **The statistical-mechanics correspondence is exact only for potential games.** In an exact potential game (Monderer–Shapley), logit/Glauber dynamics are reversible with stationary Gibbs measure π(x) ∝ exp(βΦ(x)); Nash = zero-temperature limit. Generic oligopoly pricing games are NOT potential games, so their logit dynamics break detailed balance and settle into a non-equilibrium steady state (NESS) with non-zero probability current — this is where Edgeworth-style price cycles live, and where entropy production is a meaningful, measurable observable.

3. **QRSE (Scharfenaker–Foley) is the closest existing econophysics analogue and must be positioned against.** It is a constrained-MaxEnt model with a logit "quantal response" action conditional plus a negative-feedback constraint; its limitations (underdetermination, arbitrariness of constraints, equilibrium-only applicability) are partly self-acknowledged in Scharfenaker & Foley (2017).

4. **The dispersion–response estimator of λ is the flagship novel contribution**, theoretically underwritten by the FDT identity; it must be validated against MLE and against a CCP-style estimator borrowed from dynamic discrete choice (Hotz–Miller), which may allow λ to be estimated WITHOUT solving the equilibrium.

5. **Solver recommendation: entropy-regularised first-order methods (magnetic mirror descent) with implicit differentiation, in JAX**, with Gambit's logit homotopy as the ground-truth oracle for small games.

---

## Details

### A. QRE foundations and the full literature

**Founding definitions.** McKelvey and Palfrey, "Quantal Response Equilibria for Normal Form Games," *Games and Economic Behavior* 10(1):6–38 (1995) define a QRE as a fixed point σ = Q(U(σ)) of the quantal-response map; existence follows from Brouwer. For the logit specification,
σ_i(a) = exp(λ U_i(a;σ)) / Σ_b exp(λ U_i(b;σ)).
As λ→∞, the logit solution branch (principal branch) traces a unique path from the centroid (λ=0) selecting a specific Nash equilibrium generically. Extended to extensive form as Agent QRE (AQRE) in McKelvey and Palfrey, "Quantal Response Equilibria for Extensive Form Games," *Experimental Economics* 1:9–41 (1998).

**Computation of the correspondence.** Turocy, "A dynamic homotopy interpretation of the logistic quantal response equilibrium correspondence," *Games and Economic Behavior* 51(2):243–263 (2005) gives the predictor-corrector path-following method implemented in Gambit; Turocy (2010), "Computing sequential equilibria using agent quantal response equilibria," *Economic Theory* 42(1):255–269 uses the same to compute Nash/sequential equilibria.

**Regular QRE and the axioms.** Goeree, Holt, and Palfrey, "Regular Quantal Response Equilibrium," *Experimental Economics* 8:347–367 (2005) introduce the axioms — interiority, continuity, responsiveness, monotonicity — that a structural quantal response function must satisfy; consolidated in their book *Quantal Response Equilibrium: A Stochastic Theory of Games*, Princeton University Press (2016).

**The empirical-content critique — the central threat.** Haile, Hortaçsu, and Kosenok, "On the Empirical Content of Quantal Response Equilibrium," *American Economic Review* 98(1):180–200 (2008): without restrictions on the quantal response functions, QRE can rationalise ANY distribution of behaviour in ANY normal-form game — it imposes no falsifiable restrictions. The regular-QRE axioms are the GHP response; the programme must confront this head-on (see Recommendations, Stage 2).

**Behavioural horse-races.** Wright and Leyton-Brown, "Predicting Human Behavior in Unrepeated, Simultaneous-Move Games," *Games and Economic Behavior* 106:16–37 (2017), a meta-analysis of five models (QRE, Level-k, Cognitive Hierarchy, QLk, Noisy Introspection) across ten different data sets, found that the QLk (quantal level-k) model consistently achieved the best performance — i.e., pure QRE is bettered by hybrids that add non-equilibrium beliefs. This is a caution: the firm-side application should treat logit-QRE as a baseline, not a presumptive winner.

**Dynamic foundations.** Logit dynamics (Blume 1993) induce ergodic Markov chains whose stationary distribution is meaningful for boundedly rational agents; stochastic fictitious play converges to logit QRE (Hofbauer–Sandholm); potential-game connections are in Section C.

### B. The MaxEnt / entropy-regularisation bridge

**Exact variational identity.** The logit response is exactly
σ* = argmax_σ [ E_σ(U) + (1/λ) H(σ) ],
with H the Shannon entropy — the Gibbs variational principle with free energy F = U − T·S, T = 1/λ. Legendre duality between entropy and the log-partition function gives the cumulant structure. The **Williams–Daly–Zachary theorem** (McFadden 1978, 1981; Williams 1977; Daly–Zachary 1978) states ∇ν(c) = p(c): the gradient of the social-surplus/logsum equals the choice probabilities; the exponential-family CGF property then makes the Hessian equal to the choice-indicator covariance. Together these are the discrete-choice FDT.

**Rational inattention micro-foundation for λ.** Matějka and McKay, "Rational Inattention to Discrete Choices: A New Foundation for the Multinomial Logit Model," *American Economic Review* 105(1):272–298 (2015): a Shannon-mutual-information cost of attention generates exactly the (generalised) multinomial logit, with λ the inverse information price. Generalised to arbitrary additive-random-utility forms via Bregman information costs by Fosgerau, Melo, de Palma, and Shum, "Discrete Choice and Rational Inattention: A General Equivalence Result," *International Economic Review* 61(4) (2020). This directly addresses the user's concern that λ is "a garbage parameter" — RI decomposes it into an information cost.

**Perturbed utility / convex-conjugate view.** Fudenberg, Iijima, and Strzalecki, "Stochastic Choice and Revealed Perturbed Utility," *Econometrica* 83(6):2371–2409 (2015): stochastic choice = expected utility plus a nonlinear (convex) perturbation; logit is the Shannon-entropy special case. General convex regularisers (Tsallis entropy, α-entmax) produce sparse-support responses (sparsemax/entmax) — relevant because real price grids concentrate on a few price points (9-endings), so a sparse quantal response may fit better than logit.

**Thermodynamics of decision-making.** Wolpert, "Information Theory — The Bridge Connecting Bounded Rational Game Theory and Statistical Physics" (2004/2006), Product Distribution theory; Ortega and Braun, "Thermodynamics as a theory of decision-making with information-processing costs," *Proceedings of the Royal Society A* 469(2153):20120683 (2013; arXiv:1204.6481), whose free-energy functional trades off internal energy (utility proxy) against a KL/mutual-information cost, with β the price of computation and MNL recovered as a special case.

**Solver-relevant equilibrium results.** Sokota et al., "A Unified Approach to Reinforcement Learning, Quantal Response Equilibria, and Two-Player Zero-Sum Games," ICLR 2023 (arXiv:2206.05825): magnetic mirror descent (MMD) converges linearly (last-iterate) to QRE = entropy-regularised Nash in extensive-form games — the first first-order solver with this guarantee; implemented in OpenSpiel. Related last-iterate work: Cen–Wei–Chi; adaptively perturbed mirror descent (Abe et al.).

### C. Statistical mechanics and non-equilibrium thermodynamics of pricing games

**Boltzmann/Gibbs dictionary (exact for potential games).** With λ = β, negative expected payoff = energy, Z = partition function. Monderer–Shapley potential games: logit dynamics are reversible (equivalent to Glauber dynamics) with stationary Gibbs measure π(x) ∝ exp(βΦ(x)), Φ the potential; Nash = β→∞ limit. This is an exact equivalence — the detailed-balance case (established in the logit-dynamics/Gibbs-measure literature, e.g. Blume 1993; Alós-Ferrer–Netzer; the near-potential-games work of Candogan et al.).

**Brock–Durlauf as the mean-field Ising analogue.** Brock and Durlauf, "Discrete Choice with Social Interactions," *Review of Economic Studies* 68(2):235–260 (2001): logit choice with a conformity term is literally mean-field (Curie–Weiss) Ising; multiple self-consistent equilibria appear when the interaction strength exceeds a threshold — a phase transition. This is the template for the phase diagram of the pricing game (Section E). Bouchaud's random-field-Ising extension and Kirman's Marseille-fish-market herding model are the natural generalisations.

**Non-equilibrium regime — the substantive novelty.** Generic oligopoly pricing is NOT a potential game (best responses do not derive from a common potential), so logit dynamics break detailed balance → NESS with non-zero probability current J*. Edgeworth price cycles are exactly a broken-detailed-balance limit cycle. Xu & Wang, "Measurement and Application of Entropy Production Rate in Human Subject Social Interaction Systems" (arXiv:1107.6043) — which states the entropy production rate "has never been reported empirically" before their work — show both how to test the minimax randomization model with experimental 2×2 games data and how to identify the Edgeworth price cycle in experimental market data via a non-zero EPR. The stochastic-thermodynamics machinery (Jarzynski equality, Crooks fluctuation theorem, thermodynamic uncertainty relation, Hatano–Sasa excess/housekeeping decomposition, Onsager reciprocity) then bounds the precision of price cycles: the TUR gives a lower bound on the entropy production required to sustain a cycle of given regularity.

**QRSE.** Scharfenaker and Foley, "Quantal Response Statistical Equilibrium in Economic Interactions: Theory and Estimation," *Entropy* 19(9):444 (2017): constrained MaxEnt over the joint action–outcome space with a logit action conditional and a negative-feedback (competition) constraint, applied to profit-rate distributions and estimated by Bayesian methods; developed in Scharfenaker (2020, "Implications of Quantal Response Statistical Equilibrium," *J. Economic Dynamics and Control* 119). Limitations, partly self-acknowledged in the 2017 *Entropy* paper: underdetermination/non-identification of individual dynamics ("the constraints… do not determine any unique individual dynamics"), arbitrariness of the constraint set, and equilibrium-only applicability (no out-of-equilibrium/irreversible dynamics). For a hostile external attack a referee would reach for the philosophy-of-MaxEnt literature (Uffink 1995; Seidenfeld 1986).

### D. Demand, elasticity, and the pricing literature underneath

**Structural demand.** Berry, Levinsohn, and Pakes (1995) random-coefficients logit (BLP); implemented in pyblp (Conlon and Gortmaker, "Best practices for differentiated products demand estimation with PyBLP," *RAND Journal of Economics* 51(4):1108–1161, 2020). The key conceptual point: BLP's consumer-side logit and QRE's firm-side logit are the same mathematics at different layers — price endogeneity (Hausman, BLP, differentiation IVs) reappears on the firm side when estimating λ.

**Conduct testing = λ estimation.** Duarte, Magnolfi, Sølvsten, and Sullivan, "Testing Firm Conduct," *Quantitative Economics* 15(3):571–606 (2024), with the pyRVtest package, use Rivers–Vuong model selection on Berry–Haile falsifiable restrictions and provide a weak-instruments diagnostic for testing. Estimating λ IS a conduct-testing problem: λ indexes a family of conduct models between Nash-Bertrand (λ→∞) and uniform randomisation (λ→0).

**Dominick's elasticities.** Gandhi, Lu & Shi, "Estimating Demand for Differentiated Products with Zeroes in Market Share Data," *Quantitative Economics* 14(2):381–418 (2023), applied to Dominick's Finer Foods tuna, show that controlling for products with zero demand "gives demand estimates that can be more than twice as elastic than standard estimates that select out the zeroes" — elasticities roughly double when zeroes are properly handled, and rise further in high-demand periods (Lent). Measurement error in shares has a similar doubling effect (error-in-shares work on Dominick's). Toro-González et al. (2014) report highly inelastic own-price elasticities (≈ −0.13 to −0.22) for beer categories in Dominick's. Implication: category- and treatment-of-zeros choices swamp the signal, so the demand stage must be done carefully before any λ inference.

**Algorithmic pricing / collusion.** Calvano, Calzolari, Denicolò, and Pastorello, "Artificial Intelligence, Algorithmic Pricing, and Collusion," *American Economic Review* 110(10):3267–3297 (2020): Q-learning agents learn supracompetitive prices sustained by punishment strategies, without communication; their 2021 *IJIO* paper extends this to imperfect monitoring (Green–Porter environment). The empirical German-gasoline evidence is Assad, Clark, Ershov, Xu. The identification worry: low-λ QRE (noisy competition) and collusion can both raise average price and dispersion — the programme must separate "noise" from "collusion."

### E. Computation and solvers

- **Ground-truth oracle:** Gambit logit homotopy (Turocy) for small games, tracing the principal branch and handling turning points/bifurcations via predictor-corrector continuation.
- **Scalable solver:** magnetic mirror descent / entropy-regularised mirror descent in JAX; last-iterate linear convergence to QRE. Fallbacks: damped/Anderson-accelerated fixed-point iteration, optimistic gradient, Follow-the-Regularised-Leader.
- **Contraction condition:** the logit best-response map is a contraction (unique QRE) for small λ (small β/high temperature), with explicit bounds relating λ, payoff range, and action-set size; multiplicity/phase transition appears for large λ — this is the phase boundary in λ and the same threshold as Brock–Durlauf multiplicity.
- **Continuous price space:** integral fixed-point equation σ(p) ∝ exp(λ U[p;σ]); mean-field-game / McKean–Vlasov / Fokker–Planck formulation; Gaussian-process or spectral representation of the equilibrium density with controlled discretisation error.
- **Estimation machinery:** Rust NFXP vs Su–Judd MPEC (*Econometrica* 80, 2012; Dubé–Fox–Su for BLP) vs Hotz–Miller / Aguirregabiria–Mira two-step CCP. The CCP insight — invert observed choice probabilities to recover choice-specific values/payoffs without solving the equilibrium — is the key to scaling λ-estimation to Dominick's; estimating QRE is structurally identical to estimating a dynamic discrete-choice model.
- **Autodiff/implicit differentiation:** differentiate through the QRE fixed point (implicit function theorem, deep-equilibrium-model style), jaxopt/optax implicit diff. Recommend **JAX** over PyTorch for the fixed-point + implicit-diff stack; **pyblp** for demand; **Gambit** Python API and **OpenSpiel** (which ships MMD) for equilibrium; **nashpy** only for toy checks. Honest maturity note: pyblp and Gambit are production-grade; a general continuous-price QRE solver does not exist off-the-shelf and is a genuine deliverable.
- **Bayesian hierarchical λ:** partial pooling across stores/zones/categories in NumPyro/Stan, identifying heterogeneous λ_{store,category}.

### F. The dreamprice and Dominick's data

**dreamprice on Hugging Face.** Exact repo id: **`qbz506/dreamprice-dominicks-cso`**. It is a processed Dominick's canned-soup (CSO) slice: 500,000 rows (train 371.5k / validation 84.7k / test 43.8k), **529 columns**, Parquet, licence **CC-BY-NC-4.0**, tags include tabular-regression, time-series-forecasting, demand-estimation, scanner-data; the card states it is "used for training the DreamPrice world model for retail pricing optimization" and that the source is the Kilts Center Dominick's database. It merges:
- **Movement variables:** STORE, UPC, WEEK, MOVE (units sold), QTY (bundle size), PRICE, SALE (∈{B,C,S}, promo flag), PROFIT (% margin), OK (validity flag), plus full-precision PRICE_HEX / PROFIT_HEX.
- **Store demographics (the full `account` file):** AGE9, AGE60, ETHNIC, EDUC, INCOME, GINI, HSIZE*, HVAL*, POVERTY, SHOP* … SHOPINDX.
- **Store/zone identifiers:** NAME, CITY, ZIP, LAT, LONG, ZONE (price zone), SCLUSTER, WEEKVOL.
- **Store-level elasticity fields:** the SELAS/SEELAS, UELAS/WELAS/NELAS/MELAS families and SELASALL (these are precomputed elasticity/competition fields from the Kilts store file).
- **UPC descriptors:** COM_CODE, DESCRIP, SIZE, CASE, NITEM.
- **Engineered fields:** unit_price, log_price, log_move, and **hausman_iv** (a ready-made Hausman-style instrument).

Sample rows confirm PRICE in dollars (e.g., 0.99, 0.89, 1.31), MOVE as integer units, QTY=1 for singletons, PROFIT as percent. **No other "dreamprice"/"dream-price"/"DreamPrice" dataset was found on the Hub** — this is the match; the only near neighbour in a general search is `Rif-SQL/time-series-uk-retail-supermarket-price-data`, which is unrelated UK data.

**Canonical Dominick's structure (Kilts Center manual).** Weekly store-level scanner data, ~100-store chain (Chicago metro), Sept 1989–May 1997, ~29 category acronyms (ana, ber, bjc, cer, che, cig, coo, cra, **cso**, did, …, tna, tpa, tti). File types:
- **Customer Count** (`ccount`): daily, by DFF department (bakery, deli, produce…), store traffic and coupons.
- **Store Demographics** (`account`): 1990 census-derived per store.
- **UPC files** (`upcxxx`): upc, com_code, nitem, descrip, size, case (last item-code digit: 0 = drop-shipped, 1 = warehoused).
- **Movement files** (`wxxx`): upc, store, week, move, price, qty, profit, sale, ok, sorted by upc/store/week.

Known quirks the programme must handle:
- **The qty bundle gotcha:** for bundle offers (e.g., "3 for $2"), PRICE is the bundle price and QTY the bundle size, while MOVE is units (not bundles), so **Sales = Price·Move/Qty**.
- **PROFIT** is percent gross margin on **average acquisition cost (AAC)**, not replacement cost — AAC adjusts sluggishly and drops precipitously on forward-buying (Peltzman); economically "wrong" cost basis.
- **SALE flag** is inconsistently set (unset ≠ no promotion).
- **OK=0** flags suspect weeks (drop them).
- **16 price zones collapsing to four tiers** (Cub-Fighter, low, medium, high); some post-1992 stores lack a zone. Regular prices uniform within zone; promoted prices chain-wide.
- **Week decode** table maps DB week → calendar week with special-events flags (holidays), essential for de-seasonalising.

**Mapping to model objects.**
- **Price grid** 𝒜 = empirical support of PRICE per UPC within a zone (this is the discrete action set; expect concentration at 9-endings → motivates sparse/entmax responses).
- **Action distribution** σ = cross-store/cross-week histogram of prices within a zone-week cell.
- **Expected payoff** U(p) = per-unit margin × volume, reconstructed from PROFIT (margin) and MOVE (with the qty correction).
- **FDT fluctuation** = within-zone (or within-store-over-weeks) price variance Var(p); its conjugate response is the estimated own-price elasticity from the demand stage.
- **Competitor prices** proxied via the zone structure and the store-file competition fields (COMPCUB, COMPNEAR, DTCOMP…); a genuine rival-firm interpretation is a modelling assumption (see Caveats).
- **Instrument:** hausman_iv is provided; Hausman/BLP/differentiation IVs available from cross-zone price variation.

---

## Recommendations

1. **Stage 1 (months 0–3): reproduce the exact identities and the FDT result.** Implement logit-QRE for a small discrete-price duopoly in JAX; verify against Gambit's logit homotopy. Derive and numerically confirm the firm-side static FDT — that the derivative of expected chosen price with respect to a payoff perturbation equals λ times the variance of the equilibrium price distribution (the Hessian-of-logsum = covariance identity), extending Garnier-Brun–Bouchaud–Benzaquen's Slutsky-matrix fluctuation-response formula to the firm side. *Threshold to change course:* if the analytically derived susceptibility does not match the numerically computed one to machine precision on the potential-game case, the identity is mis-stated — stop and re-derive before touching data.

2. **Stage 2 (months 3–9): estimate demand on `qbz506/dreamprice-dominicks-cso` (and raw CSO movement/UPC files) with pyblp**, producing payoffs U that pin down the game. This is what restores empirical content in the face of the Haile–Hortaçsu–Kosenok theorem — QRE regains falsifiable content once the payoff matrix is externally identified. *Threshold:* own-price elasticities must be negative and stable across specifications; given Gandhi–Lu–Shi, run the estimation with and without zero-share correction and expect up to a doubling — if elasticities are not robust, the λ stage is not yet trustworthy.

3. **Stage 3 (months 9–15): estimate λ three ways** — MLE via NFXP, the **dispersion–response (FDT) estimator**, and a Hotz–Miller CCP two-step (which sidesteps solving the equilibrium). *Falsification criterion:* the three estimates should coincide within confidence bounds. Systematic divergence of the dispersion estimator from the MLE is itself the diagnostic that detailed balance is broken (non-potential structure / cycling), which is a finding, not a failure.

4. **Stage 4 (months 15–24): map the phase diagram** in (λ, number of firms N, cross-price elasticity) — locating the contraction/uniqueness region (small λ, diffuse pricing), the multiplicity/coordination region (large λ, à la Brock–Durlauf), and the cycling region — and **measure the entropy production rate** (à la Xu–Wang) to distinguish an Edgeworth-type NESS from noisy static competition. Frame λ explicitly as a conduct parameter and run pyRVtest-style Rivers–Vuong model selection against Nash-Bertrand and collusive alternatives, with the weak-instrument diagnostic.

5. **Publish the FDT / "elasticity IS a susceptibility, price dispersion IS its conjugate fluctuation" result first** — it is the cleanest genuinely new contribution and does not depend on the messier λ-identification questions.

---

## Caveats

- **The physics is exact only for potential games.** For generic pricing games the global Gibbs-measure statement fails; what survives is (a) the exact single-agent variational/entropy-regularisation identity and (b) the NESS framing. State this sharply — do not claim a global Hamiltonian/potential exists where it does not. This is the distinction between an *exact equivalence* (potential-game Gibbs measure; single-player logit = entropy-regularised best response; logsum Hessian = covariance) and a *useful formal analogy* (β = λ, energy = −payoff in non-potential games, where no unique potential exists). "Temperature" as a global control knob is decorative once detailed balance breaks.
- **HHK non-falsifiability is only defused by the separately-estimated demand system.** If demand is mis-specified, λ absorbs the mis-specification — exactly the "garbage parameter" risk. Rational inattention gives λ a structural interpretation (information price) but does not by itself guarantee identification.
- **λ vs collusion identification is genuinely hard.** Low λ (noisy competition) and collusion are observationally similar in both price level and dispersion; the conduct-test and entropy-production diagnostics are the proposed separators, but this remains the highest-risk assumption.
- **Dominick's is 1989–1997, a single chain in one metro.** External validity to modern CPG retail is limited; the "competitor price" is largely within-chain across zones, not true rival firms, so the oligopoly interpretation is a modelling assumption, not a data feature. The programme's claim to generality rests on the solver being domain-agnostic, not on Dominick's being a clean oligopoly.
- **The dispersion–response estimator assumes stationarity.** Promotions, holidays (the week-decode special events), and the AAC accounting lag inject non-stationarity that will bias a naïve within-cell variance; de-seasonalise and drop OK=0 weeks first.
- **The sparse-response (α-entmax) advantage for 9-ending price points is a hypothesis, not a result** — worth testing but not to be assumed.
- **A hostile referee's strongest lines** will be: (i) "QRE has no empirical content" (answer: two-stage design with external demand); (ii) "your thermodynamics is metaphor" (answer: concede for non-potential games, stand firm on the exact FDT/CGF identity and the measurable entropy production); (iii) "λ is unidentified from collusion" (answer: conduct test + EPR, and honest acknowledgement it is partial); (iv) "QRSE already did MaxEnt economics" (answer: QRSE is aggregate-outcome MaxEnt with acknowledged non-identification; this programme is game-theoretic, payoff-identified, and adds the non-equilibrium/current structure QRSE explicitly cannot represent).