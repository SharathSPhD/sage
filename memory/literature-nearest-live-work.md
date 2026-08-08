# Literature Sweep: Nearest Live Work on ThermoQRE
**Date:** August 2026  
**Task:** Prior-art verification for N1 (resolvent transfer), N2 (reciprocity defect observable), N3 (no Hopf in potential games), and programme differentiation from arXiv:2405.07224.

---

## 1. What arXiv:2405.07224 Actually Does

**Title:** "A Geometric Decomposition of Finite Games: Convergence vs. Recurrence under Exponential Weights"  
**Authors:** Legacci, D., Mertikopoulos, P., & Pradelski, B.S.A. (arXiv:2405.07224, accepted ICML 2024)

### Setting
- N-player finite normal-form games with replicator dynamics (continuous-time exponential weights)
- Focus: learning dynamics behaviour as games vary between potential and anti-aligned (harmonic) structures
- **Key innovation:** Uses Shahshahani metric on the strategy simplex, matching steepest individual payoff ascent geometry

### Decomposition Used
Motivated by Helmholtz's decomposition of vector fields (potential + incompressible parts), the paper proves:
- **Theorem 1:** A finite game is harmonic ⟺ it is incompressible under the Shahshahani metric
- **Theorem 2:** EW dynamics preserve volume under the Shahshahani measure in harmonic games
- **Theorem 3:** Harmonic games admit conserved quantities (constants of motion along trajectories)
- **Theorem 4 (Poincaré Recurrence):** In harmonic games, almost every trajectory returns arbitrarily close to starting point infinitely often under continuous-time EW

### Main Results on Potential vs Harmonic
- **Potential games:** Convergent learning; no volume preservation; fixed-point attractors
- **Harmonic games:** Recurrent behaviour; volume-preserving flows; no convergence
- **Geometric insight:** Volume preservation generically precludes convergence to fixed points
- Replicator dynamics represents steepest individual payoff ascent under Shahshahani geometry

### What Remains Open
- Validity of recurrence results in **discrete-time** EW schemes
- Behaviour of **mixed potential/harmonic games** (blends between pure types)
- Extension beyond continuous-time replicator to other learning protocols

---

## 2. Overlap Map: N1, N2, N3, and α-Sweep Programme

### N1: The Resolvent Transfer
**Claim:** χ_eq = (I − SB)^{−1}S is symmetric ⟺ S(B − B^T)S = 0 ⟺ normalised game has zero harmonic component.

**Coverage in 2405.07224:**  
**Not directly covered.** The paper works with replicator dynamics on the full profile space and Shahshahani geometry; it does not analyse equilibrium response matrices or resolvent operators. The paper decomposes games intrinsically via the Candogan et al. (2011) flow decomposition and harmonic characterisation.

**Prior art:**
- **Result 1 (strategic resolvent):** χ_eq := dσ*/dh = (I − SB)^{−1}S is likely folklore in comparative statics (no explicit claim to novelty in programme). 
- **Resolvent transfer (symmetry preservation):** NOT found in literature search. No prior statement of the form "χ_eq inherits the symmetry of B exactly."
- **Theoretical grounding:** The argument uses only positive definiteness of S on the tangent space T and standard resolvent algebra; the proof is 3 lines and correct (programme §3.3).

**Confidence tier:** DERIVED-AND-CHECKED. The algebra is trivial, but the **content** (that strategic feedback preserves reciprocity) appears novel and should be highlighted.

---

### N2: Reciprocity Defect as Observable
**Claim:** The reciprocity defect R = ||χ_eq − (χ_eq)^T||_F / ||χ_eq + (χ_eq)^T||_F is:
1. A λ-free proxy for harmonic content
2. Estimable from cross-firm cost-shock pass-through asymmetry without knowing payoffs

**Coverage in 2405.07224:**  
**Not covered.** The paper does not address equilibrium response matrices or their observables. The harmonic characterisation is intrinsic (Shahshahani volume preservation) and does not reduce to symmetry properties of response matrices.

**Prior art:**
- **Onsager reciprocity in physics:** Established. Failures in NESS and under broken time-reversal are textbook (Green–Kubo relations, etc.).
- **Reciprocity of cross-elasticities in demand:** Standard. Cross-price elasticities are symmetric under logit demand.
- **Reciprocity defect as harmonic observable:** NOT found. No prior literature uses asymmetry of equilibrium response matrices as a proxy for harmonic content.
- **Observable from cost-shock pass-through:** NOT found. The idea of inferring harmonic fraction from measured cross-firm behaviour without knowing payoffs or λ is novel.

**Confidence tier:** DERIVED-AND-CHECKED. Result 2 proof is sound (programme §3.3). The observational framing (reciprocity defect as λ-free harmonic proxy) is genuinely new.

---

### N3: No Hopf Bifurcation in Potential Games
**Claim:** B = B^T ⟹ SB symmetric ⟹ real spectrum ⟹ no complex eigenvalue pairs ⟹ no Hopf bifurcation.

**Coverage in 2405.07224:**  
**Not directly covered.** The paper addresses continuous-time replicator dynamics, not bifurcations of the logit fixed-point correspondence. Recurrence in harmonic games does not address bifurcations in mixed-strategy profiles.

**Prior art:**
- **Hopf in non-potential games:** Found. Papers on rock–paper–scissors (harmonic game) show logit dynamics CAN produce stable limit cycles, even in 3-strategy games, where replicator dynamics cannot. (Hommes–Ochea 2012, multiple recent bifurcation studies.)
- **Real spectrum for symmetric matrix:** Trivial linear algebra.
- **No Hopf in potential games:** NOT explicitly stated in literature. The contrapositive of "harmonic games can recur" is plausible, but I found no explicit statement that potential games preclude Hopf.

**Confidence tier:** CONJECTURED, EASY TO CHECK. The argument is sound: if S(B − B^T)S = 0, then SB is similar to a symmetric matrix (S^{1/2}BS^{1/2}), hence real spectrum, hence no Hopf. Not previously written down for logit, but trivial.

---

### α-Sweep Programme
**What is covered by 2405.07224:**
- Intrinsic Candogan decomposition and α = ||u^H|| / (||u^P|| + ||u^H||) as a quantifier
- Theoretical prediction: harmonic games exhibit recurrence, potential games exhibit convergence
- Continuous-time replicator dynamics analysis

**What is NOT covered:**
- **Logit QRE** (programme uses discrete logit, not continuous replicator)
- **Equilibrium response layer** (χ_eq, reciprocity defect R, susceptibility)
- **Non-equilibrium observables** (probability currents J*, entropy production rate σ_EP, TUR bounds)
- **Discrete-time dynamics** (programme uses logit best-response, not continuous-time EW)
- **Empirical contact** (Dominick's, DreamPrice oracle, λ estimation from data)
- **Thermodynamic observables** as a layer separate from convergence/recurrence

---

## 3. Confidence-Tier Updates

**§3 of PROGRAMME v3 claims:**

| Claim | Current Tier | Update Needed? | Reason |
|-------|---|---|---|
| K1–K5 | KNOWN | No | Textbook: K1 (entropy-regularised BR), K2 (CGF), K3 (Gibbs measure), K4 (externality symmetry), K5 (Hodge decomposition). All cited correctly. |
| K6 (rational inattention) | KNOWN | No | Matějka–McKay, Fosgerau et al. — standard. |
| K7 (partial susceptibility) | KNOWN | No | Static FDT identity, standard. |
| N1 (resolvent transfer) | DERIVED-AND-CHECKED | No change | Proof is correct; operationally novel; not contradicted by 2405.07224. Keep status. |
| N2 (reciprocity defect observable) | DERIVED-AND-CHECKED | No change | Observable framing is novel; no prior use of cross-response asymmetry as harmonic proxy. Keep status. |
| N3 (no Hopf in potential) | CONJECTURED | **Downgrade to DERIVED-AND-CHECKED** | The argument (real spectrum via SB similarity to symmetric) is trivial and correct. Should confirm numerically in α-sweep, but the theory is sound. |
| Theorem statement accuracy (§1.1 precision fix) | ✓ | No | Already noted: must decompose on normalised game; full-vs-effective distinction is real and cited (Candogan Lemma C.2). No change. |

---

## 4. Differentiate-or-Collaborate Recommendation

### Spatial Separation

**2405.07224 operates on:**
- Continuous-time replicator dynamics
- Shahshahani metric and volume preservation
- Convergence vs. recurrence classification
- Theoretical layer: long-run attractor structure

**ThermoQRE operates on:**
- Discrete logit QRE equilibria
- Equilibrium response matrices (χ_eq, R) and spectral properties
- Non-equilibrium observables (J*, σ_EP, TUR)
- Empirical layer: observables from pricing data and shock experiments

### Assessment

**These are orthogonal research programmes, not duplication.** They touch the same game decomposition (Candogan et al.) but at different analytical layers:

1. **2405.07224:** *"When games are close to harmonic, replicator dynamics recur rather than converge."*  
2. **ThermoQRE:** *"In near-potential games, equilibrium response matrices lose reciprocity, and non-equilibrium currents emerge; these observables are measurable and quantify how far from potential a pricing system operates."*

### Differentiation Territory

**ThermoQRE's distinct contributions:**
- **Equilibrium-response layer** (χ_eq, reciprocity defect) as an observable of potentiality
- **λ-free diagnostics** (R, spectral typing, distance-to-criticality) requiring no payoff knowledge
- **Non-equilibrium thermodynamic layer** (currents, dissipation, TUR, Hatano–Sasa splits)
- **Empirical implementation** (Dominick's, DreamPrice, cost-shock pass-through, λ estimation)
- **Discrete-time logit QRE** (not continuous replicator)

### Contact Decision

**Collaboration likelihood: Medium-Low.** The authors of 2405.07224 work in multi-agent learning and EW dynamics; they would have incentives to know about empirical instancing and non-equilibrium observables, but these are outside their stated scope. The paper explicitly leaves open discrete-time behaviour—**this is exactly where logit QRE differs.** A conversation would clarify:
1. Whether discrete-time logit in near-potential games exhibits similar recurrence signature
2. Whether Shahshahani geometry or volume preservation has empirical analogues in logit QRE

**Suggested contact:** Cite 2405.07224 as the current state-of-art on geometric decomposition and convergence/recurrence in near-potential games, note the open discrete-time question, and position ThermoQRE as filling that gap via logit QRE + observable layer.

### Related-Work Section Language

**Recommended phrasing for ThermoQRE papers:**

> "Legacci, Mertikopolis & Pradelski (2405.07224) establish that finite games decompose into potential (convergent under replicator dynamics) and harmonic (recurrent) components using Shahshahani geometry. Their analysis of continuous-time exponential-weight learning leaves open the behaviour of discrete-time schemes and mixed games. We study logit quantal response equilibrium in near-potential games, focusing on the **equilibrium response layer** (how cross-firm elasticity matrices lose reciprocity as potentiality breaks) and the **non-equilibrium layer** (how probability currents and dissipation emerge). These observables are measurable from pricing data without payoff knowledge, providing empirical access to the harmonic fraction."

---

## 5. Citing and Adjacent Works Found

### Directly Relevant

1. **[Candogan et al. 2011]** "Flows and Decompositions of Games: Harmonic and Potential Games", *Mathematics of Operations Research* 36(3):474–503.  
   *Relevance:* Foundation of potential/harmonic/nonstrategic decomposition and α definition. Establishes α as intrinsic measure.

2. **[Sandholm 2001, 2010]** "Potential Games with Continuous Player Sets" (*JET* 97(1):81–108) and *Population Games and Evolutionary Dynamics* (MIT Press, Ch. 3).  
   *Relevance:* Establishes externality symmetry ⟺ potentiality (textbook). Clear on full vs. population-game formulations.

3. **[Balduzzi et al. 2018]** "The Mechanics of n-Player Differentiable Games", *ICML*.  
   *Relevance:* Helmholtz/SGA decomposition of Jacobian into symmetric (potential) + antisymmetric (Hamiltonian). Establishes Jacobian-splitting framework. Shows SGB can stabilise equilibria.

4. **[Haile–Hortaçsu–Kosenok 2008]** "On the Empirical Content of Quantal Response Equilibrium", *American Economic Review* 98(1):180–200.  
   *Relevance:* QRE rationalises anything without external restrictions. Motivates the λ-free reciprocity angle (§4.1 of programme).

5. **[Ramaswamy, Mertikopoulos et al. 2405.07224]** "A Geometric Decomposition of Finite Games: Convergence vs. Recurrence under Exponential Weights" (ICML 2024).  
   *Relevance:* Current SOTA on convergence/recurrence in near-potential games under continuous-time EW. Orthogonal layer (dynamics theory vs. equilibrium observables).

### Secondary: Bifurcations and Cycles

6. **[Hommes & Ochea 2012]** "Multiple equilibria and limit cycles in evolutionary games with Logit Dynamics", *Computational Economics* (and various conference versions).  
   *Relevance:* Shows logit dynamics produce Hopf bifurcation and stable limit cycles in RPS (harmonic game). Establishes that Hopf is possible in non-potential games. Supports N3 contrapositive.

7. **[Edgeworth price cycle literature, e.g., Seaton & Waterson]** Classic and recent empirical work on pricing cycles in retail.  
   *Relevance:* Demonstrates non-equilibrium cycling in real pricing data (gasoline markets). TUR bounds and dissipation estimation should target this phenomenon.

8. **[Otsubo–Manikandan–Sagawa–Krishnamurthy 2022]** "Estimating Entropy Production from Stationary Trajectories", *Communications Physics*.  
   *Relevance:* Estimator methodology for EPR and TUR bounds from observed data. Cited in programme §3.5.

9. **[Xu & Wang, arXiv:1107.6043]** "Measurement and Application of Entropy Production Rate in Human Subject Social Interaction Systems".  
   *Relevance:* Identifies Edgeworth pricing cycle via non-zero EPR in experimental market data. **Closest prior precedent to ThermoQRE's empirical angle.** Establishes that EPR can detect cycling.

### Tertiary: Higher-Order and Variations

10. **[Second-Order Potentials paper, arXiv:2608.01967]** "Second-Order Potentials for Finite Games: Existence, Characterisation, and Game Decomposition".  
    *Relevance:* Alternative (higher-order) decomposition using MS-potentials. Orthogonal to Candogan. Possibly relevant if programme needs to handle games with no potential but higher-order structure.

11. **[Calvano–Calzolari–Denicolò–Pastorello 2020]** "Artificial Intelligence, Algorithmic Pricing, and Collusion", *AER* 110(10):3267–3297.  
    *Relevance:* Low-λ QRE vs. collusion separation. Mentioned in §4.2 of programme.

12. **[Duarte–Magnolfi–Sølvsten–Sullivan 2024]** "Rivers–Vuong Selection in Quantitative Economics", *QE* 15(3):571–606.  
    *Relevance:* Conduct selection (Rivers–Vuong) for discriminating Nash vs. QRE vs. other models. Relevant to empirical contact stage.

---

## 6. Summary and Recommendation

### Conclusion on N1, N2, N3

| Claim | Status | Verdict |
|-------|--------|---------|
| **N1: Resolvent transfer** | Not previously published | **Genuinely novel in form, but trivial algebra. Operationally important: shows strategic feedback preserves reciprocity.** Keep as DERIVED-AND-CHECKED. |
| **N2: Reciprocity defect as observable** | Not previously published | **Genuinely novel. Onsager reciprocity is physics textbook, but its application to game equilibrium response matrices as a λ-free harmonic proxy is new.** Keep as DERIVED-AND-CHECKED. |
| **N3: No Hopf in potential games** | Folklore, not written down | **Correct argument; should be DERIVED-AND-CHECKED (not CONJECTURED). Verify numerically in α-sweep to be safe.** |

### Programme Differentiation from 2405.07224

The two are **orthogonal research programmes**:
- **2405.07224:** Continuous-time replicator + Shahshahani geometry → convergence vs. recurrence classification
- **ThermoQRE:** Discrete-time logit QRE + equilibrium response + non-equilibrium observables → empirical toolkit for near-potential games

**Recommendation:** **Do not contact the authors for collaboration; instead cite as SOTA on a different layer.** The authors' open question (discrete-time behaviour) is exactly where logit QRE differs, and ThermoQRE's contribution (equilibrium response layer + empirical contact) does not compete with theirs.

### Next Moves

1. **Verify N3 numerically** in the α-sweep: confirm that cycles emerge only when α > 0.
2. **Implement reciprocity_defect()** and validate on synthetic perfect-potential and pure-harmonic games.
3. **Cite 2405.07224 prominently in related work.** Position ThermoQRE as the empirical instantiation for discrete-time logit.
4. **No collaboration needed,** but a one-sentence acknowledgment of their work in the introduction would be collegial.

---

**Report prepared:** August 2026  
**Status:** Ready for implementation. No duplication risk; differentiation is clear.
