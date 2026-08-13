/* Permanent landmarks on the (R, sigma_EP) plane.
 *
 * Every number here was read out of the named file in benchmarks/results/ in
 * this repository. Nothing on this list is retyped from a paper draft or a brief.
 * `artifact` is the filename, `field` the key inside it, and `value` is what that
 * key holds. If a number cannot be traced that way it does not go on the plane.
 */

export type Quadrant = "I" | "II" | "III" | "IV";

export interface Provenance {
  /** File under benchmarks/results/ in this repository. */
  artifact: string;
  /** Key path inside the artifact. */
  field: string;
  /** What the artifact's own notes say the reading is and is not. */
  note: string;
}

export interface Landmark {
  id: string;
  name: string;
  kind: "real data" | "calibration anchor";
  quadrant: Quadrant;
  /** Response asymmetry R, or null where R has not been read for this system. */
  R: number | null;
  RLabel: string;
  RCi?: [number, number];
  /** Where the dissipation axis reading sits: "zero", "positive", or "untested". */
  EP: "zero" | "positive" | "untested";
  EPLabel: string;
  n?: string;
  provenance: Provenance[];
  /** One sentence a reader can act on. */
  reading: string;
}

export const LANDMARKS: Landmark[] = [
  {
    id: "sioux-falls",
    name: "Sioux Falls road network",
    kind: "calibration anchor",
    quadrant: "I",
    R: 5.648638508839636e-17,
    RLabel: "5.65 × 10⁻¹⁷",
    EP: "zero",
    EPLabel: "0 (detailed balance)",
    n: "top-8 OD pairs, k = 3 route sets",
    provenance: [
      {
        artifact: "benchmarks/results/sioux_falls_calibration.json",
        field: "metrics.reciprocity_defect = 5.648638508839636e-17",
        note: "Real TNTP Sioux Falls data. Fisk KKT spread 7.1e-15, DF-symmetry defect exactly 0. The UE link-flow gap in the same artifact is a DIAGNOSTIC ONLY — restricted route sets make exact agreement impossible by construction.",
      },
      {
        artifact: "benchmarks/results/reciprocity_potential.json",
        field: "metrics.max_R = 8.93e-17 over five verified exact potential games",
        note: "The zero end of the response axis, calibrated on games whose potential structure is known by construction.",
      },
    ],
    reading:
      "Traffic assignment is an exact potential game, so the response axis reads zero at machine precision on real network data.",
  },
  {
    id: "dominicks",
    name: "Dominick's retail panel",
    kind: "real data",
    quadrant: "I",
    R: 0.0011224257982100365,
    RLabel: "0.00112",
    RCi: [4.778562071941612e-5, 0.004993334763269472],
    EP: "zero",
    EPLabel: "at null (0 / 30 Edgeworth detections)",
    n: "22,655 store-weeks, 86 stores",
    provenance: [
      {
        artifact: "benchmarks/results/pricing_passthrough_R.json",
        field:
          "metrics.R_empirical = 0.0011224257982100365, R_ci_low = 4.778562071941612e-05, R_ci_high = 0.004993334763269472, n_store_weeks = 22655, n_stores = 86",
        note: "Cross-brand wholesale-cost pass-through, Campbell vs Progresso, regular prices only, two-way (store, week) demeaning, cluster bootstrap over stores (500 resamples). The near-symmetric prediction was written into the config before the run: one retailer prices both brands, so a single-objective optimiser must respond symmetrically.",
      },
      {
        artifact: "benchmarks/results/toolkit_verdicts.json",
        field: "metrics.f0011_r_read = 0.0011270567893383497",
        note: "The same reading reproduced through the public strataq.toolkit facade — it differs in the fifth decimal because the facade re-solves rather than reusing the cached fit.",
      },
    ],
    reading:
      "Real category pricing reads potential-like: comparative statics are trustworthy and there is no cycle to time.",
  },
  {
    id: "caiso",
    name: "CAISO SP15 day-ahead",
    kind: "real data",
    quadrant: "II",
    R: null,
    RLabel: "not read",
    EP: "positive",
    EPLabel: "0.0447 nats/hour ≈ 1.07 nats/day",
    n: "840 hourly DAM intervals, 35 days",
    provenance: [
      {
        artifact: "benchmarks/results/electricity_irreversibility_dam.json",
        field:
          "metrics.kld_embed_per_hour = 0.04465589955121747, null_markov_q99 = 0.029105357387939814, markov_detected = 1",
        note: "Phase-embedded (price bin, delta-sign) KLD against a reversibilized-Markov surrogate null with matched persistence: 0.0447 exceeds the null's q99 of 0.0291, empirical p < 0.01. Against the FT and AAFT spectral nulls the same reading sits INSIDE the band — no detection there. The detection is statistically driven by the high-ramp second half of the July-2026 window (F-0009 addendum), and it tests pair-level detailed balance of the embedding, not full-process reversibility.",
      },
      {
        artifact: "benchmarks/results/electricity_lambda.json",
        field: "metrics.model_rejected = 1, lam_hat_conditional = null",
        note: "The response axis is NOT read for this system. The stylised two-generator model whose lambda would have given it was rejected by the data: its dispersion ceiling of 10.43 $/MWh sits below the observed price std of 16.82.",
      },
    ],
    reading:
      "Something exogenous cycles the day-ahead market. Timing matters here; the structural question is still open because the response axis has not been read.",
  },
  {
    id: "rps",
    name: "Rock–paper–scissors",
    kind: "calibration anchor",
    quadrant: "IV",
    R: 0.6928203230275507,
    RLabel: "0.69 (λ = 1.2)",
    EP: "positive",
    EPLabel: "positive (NESS)",
    provenance: [
      {
        artifact: "benchmarks/results/reciprocity_harmonic.json",
        field: "metrics.R_rps_3 = 0.6928203230275507, R_rps_5 = 0.434657051228945, R_matching_pennies = 1.1999999999999995",
        note: "Read at lambda = 1.2 (config/experiments/reciprocity_calibration.yaml). R is not a fraction — matching pennies reads 1.2 — and its magnitude scales with lambda (F-0002), so only the zero/non-zero verdict is lambda-free. For this game the reading is exactly lambda / sqrt(3): 0.6928 at lambda = 1.2 and 0.8660 at lambda = 1.5, which is what the 0.69-0.87 band quoted for this anchor in DIRECTION_v4 spans. The 0.866 end is the value the deployed API returns on RPS at its own default, recorded in the web.scaffold gate.",
      },
      {
        artifact: "benchmarks/results/ness_reads_positive.json",
        field: "metrics.min_epr = 1.567826596148266, min_current = 0.08531711600773197",
        note: "Harmonic games are non-equilibrium steady states: positive dissipation and positive circulation.",
      },
    ],
    reading: "Both axes are lit. Optimising against a static model of the other side is worst exactly here.",
  },
  {
    id: "blotto",
    name: "Colonel Blotto (budget 3)",
    kind: "calibration anchor",
    quadrant: "IV",
    R: 0.11809781461174602,
    RLabel: "0.118",
    EP: "positive",
    EPLabel: "0.0983 on the asymmetric 2-field instance",
    provenance: [
      {
        artifact: "benchmarks/results/blotto_readings.json",
        field:
          "metrics.alpha_b3_k3 = 0.6939824873659469, R_b3_k3 = 0.11809781461174602, epr_b2_k2_asym = 0.0982933622271984, epr_degenerate_null = 5.392603844284259e-32",
        note: "Zero-sum does not imply harmonic-pure: alpha = 0.694 means a real ~30% potential component survives normalisation. This is the realistic high-alpha anchor; RPS is the alpha = 1 extreme.",
      },
    ],
    reading: "A zero-sum allocation game still keeps a third of its structure as a landscape.",
  },
];

/** The empty quadrant, and why it is the decisive test. */
export const QUADRANT_III = {
  name: "III. Stalled whirlpool",
  status: "no real system measured here yet",
  why:
    "Asymmetric response with no persistent circulation: one side structurally leads, but nothing cycles. If distance-from-equilibrium were a single scalar this quadrant could not exist. Finding a real system in it is the strongest available confirmation of the two-axis result, which is why it is the programme's open decisive test.",
  candidate:
    "Retail fuel pricing with asymmetric rockets-and-feathers pass-through but no Edgeworth cycling is the named candidate (DIRECTION_v4 section 5, unit R11 science.plane.quadrant_iii).",
};

export interface QuadrantSpec {
  id: Quadrant;
  name: string;
  colorVar: string;
  textVar: string;
  headline: string;
  consequence: string;
}

/** Names, colours and consequences are the ones in DIRECTION_v4 section 5. The
 *  hexes behind these variables are shared with strataq.viz PALETTE, so a
 *  reading on screen is the same colour as in the paper figures. */
export const QUADRANTS: QuadrantSpec[] = [
  {
    id: "I",
    name: "Landscape",
    colorVar: "var(--q-landscape)",
    textVar: "var(--q-landscape-text)",
    headline: "A potential game.",
    consequence:
      "Comparative statics are trustworthy, pass-through is symmetric, there are no cycles to time, and optimising against a static competitor model is correct.",
  },
  {
    id: "II",
    name: "Driven landscape",
    colorVar: "var(--q-driven)",
    textVar: "var(--q-driven-text)",
    headline: "Timing matters, structure does not.",
    consequence:
      "Reciprocal structure with circulating dynamics: something exogenous is cycling the system — demand, schedules, cost shocks — rather than the strategic interaction itself.",
  },
  {
    id: "III",
    name: "Stalled whirlpool",
    colorVar: "var(--q-stalled)",
    textVar: "var(--q-stalled-text)",
    headline: "Structure matters, timing does not.",
    consequence:
      "Asymmetric response with no persistent circulation — one agent structurally leads, but nothing cycles. Pass-through asymmetry is the exploitable object.",
  },
  {
    id: "IV",
    name: "Whirlpool",
    colorVar: "var(--q-whirlpool)",
    textVar: "var(--q-whirlpool-text)",
    headline: "Both.",
    consequence:
      "Edgeworth-cycle territory. Response asymmetry and circulation together; the regime where naive optimisation against a static rival model is worst.",
  },
];

/** The toolkit's calibrated band edges on the response axis (DIRECTION_v4 section 5). */
export const R_BANDS = { zero: 0.02, strong: 0.3 };

/** Where a fitted lambda sits against things that have been measured. */
export interface LambdaMark {
  lambda: number;
  label: string;
  detail: string;
}

export const LAMBDA_MARKS: LambdaMark[] = [
  { lambda: 0, label: "a coin", detail: "Every action equally likely: lambda = 0 is exactly uniform play." },
  {
    lambda: 1.2,
    label: "the calibration bench",
    detail:
      "The lambda every reciprocity and dynamics benchmark in this repository is read at (config/experiments/reciprocity_calibration.yaml, seed 20260808).",
  },
  {
    lambda: 4.78,
    label: "Goeree–Holt subjects",
    detail:
      "Least-squares logit fit to the three matching-pennies treatments of Goeree & Holt (2001), payoffs in dollars. Computed live on this site from their published choice frequencies.",
  },
  {
    lambda: 20,
    label: "effectively Nash",
    detail: "By lambda = 20 on payoffs of this scale the logit choice is within a percent of best response.",
  },
];
