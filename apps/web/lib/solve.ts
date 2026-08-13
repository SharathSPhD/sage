/* Turning a solved game into a decision.
 *
 * Everything here is arithmetic on top of solveQRE from lib/qre.ts, which is
 * checked against the library's goldens by scripts/test-qre.mjs. Nothing is
 * assumed, smoothed or filled in: if a quantity cannot be computed from the
 * numbers on screen, it is not shown.
 */

import { solveQRE, type Matrix } from "./qre";
import type { BuiltSituation, Situation } from "./situations";

export interface Option {
  index: number;
  label: string;
  /** Expected value against the rival distribution below. */
  value: number;
  /** Value given up versus the recommendation. Zero for the recommendation. */
  cost: number;
  /** 10th and 90th percentile of the outcome across what the rival might do. */
  lo: number;
  hi: number;
  /** Worst single rival move, and which one it is. */
  worst: number;
  worstAgainst: string;
}

export interface RivalMove {
  label: string;
  p: number;
}

export interface Flip {
  /** Which number would have to change. */
  knob: string;
  /** "above" | "below" */
  direction: "above" | "below";
  /** The value at which the answer changes, already formatted. */
  at: string;
  /** The same crossing as a number, so two candidates can be compared. */
  atValue: number;
  /** What the answer becomes. */
  becomes: string;
}

export interface Solution {
  options: Option[];
  recommended: Option;
  rival: RivalMove[];
  /** Sum of |p - uniform| — 0 when the rival is unreadable, 1 when pinned. */
  rivalConcentration: number;
  flips: Flip[];
  holds: string[];
  /** True when nothing in range flips the answer. */
  robust: boolean;
  nashOption: Option | null;
}

/** Sharpness, in payoff-spread units, at which the rival is Nash in all but name. */
const NASH_SHARPNESS = 600;

/** The rival's payoff spread — the scale that makes sharpness mean one thing. */
export function payoffSpread(built: BuiltSituation): number {
  let lo = Infinity;
  let hi = -Infinity;
  for (const row of built.u2) {
    for (const v of row) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  const span = hi - lo;
  return span > 1e-12 ? span : 1;
}

/**
 * Sharpness on screen is in payoff-spread units; the solver wants it raw.
 * One conversion, in one place, so "8" means the same thing on every page.
 */
export function lamFor(built: BuiltSituation, sharpness: number): number {
  return sharpness / payoffSpread(built);
}

/** Weighted percentile of outcomes {values[j] with weight w[j]}. */
function weightedPercentile(values: number[], w: number[], q: number): number {
  const order = values.map((v, j) => j).sort((a, b) => values[a] - values[b]);
  let acc = 0;
  for (const j of order) {
    acc += w[j];
    if (acc >= q) return values[j];
  }
  return values[order[order.length - 1]];
}

function expectedAgainst(u1: Matrix, mix: number[]): number[] {
  return u1.map((row) => row.reduce((a, v, j) => a + v * mix[j], 0));
}

function optionsFrom(built: BuiltSituation, rivalMix: number[]): Option[] {
  const ev = expectedAgainst(built.u1, rivalMix);
  const best = Math.max(...ev);
  return built.u1.map((row, i) => {
    const worstIdx = row.reduce((bi, v, j) => (v < row[bi] ? j : bi), 0);
    return {
      index: i,
      label: built.yourMoves[i].label,
      value: ev[i],
      cost: best - ev[i],
      lo: weightedPercentile(row, rivalMix, 0.1),
      hi: weightedPercentile(row, rivalMix, 0.9),
      worst: row[worstIdx],
      worstAgainst: built.theirMoves[worstIdx].label,
    };
  });
}

/** The whole answer for one set of numbers, in the browser, in under a frame. */
export function solveSituation(built: BuiltSituation, sharpness: number): Solution {
  const lam = lamFor(built, sharpness);
  const point = solveQRE({ u1: built.u1, u2: built.u2 }, lam);
  const rivalMix = point.sigma2;
  const options = optionsFrom(built, rivalMix);
  const recommended = options.reduce((a, b) => (b.value > a.value ? b : a), options[0]);

  const nashPoint = solveQRE({ u1: built.u1, u2: built.u2 }, lamFor(built, NASH_SHARPNESS));
  const nashOptions = optionsFrom(built, nashPoint.sigma2);
  const nashBest = nashOptions.reduce((a, b) => (b.value > a.value ? b : a), nashOptions[0]);
  const nashOption = nashBest.index === recommended.index ? null : options[nashBest.index];

  const n = rivalMix.length;
  const concentration =
    n <= 1 ? 1 : rivalMix.reduce((a, p) => a + Math.abs(p - 1 / n), 0) / (2 * (1 - 1 / n));

  return {
    options,
    recommended,
    rival: rivalMix.map((p, j) => ({ label: built.theirMoves[j].label, p })),
    rivalConcentration: concentration,
    flips: [],
    holds: [],
    robust: true,
    nashOption,
  };
}

/** Which action wins, for one set of parameter values. Used by the sweep. */
function bestIndex(situation: Situation, values: Record<string, number>, sharpness: number): number {
  const built = situation.build(values);
  const point = solveQRE({ u1: built.u1, u2: built.u2 }, lamFor(built, sharpness), {
    tol: 1e-8,
    maxIter: 800,
  });
  const ev = expectedAgainst(built.u1, point.sigma2);
  let bi = 0;
  for (let i = 1; i < ev.length; i++) if (ev[i] > ev[bi]) bi = i;
  return bi;
}

/**
 * What would have to change for the answer to change.
 *
 * Each knob is swept across its own range with everything else held; the first
 * crossing on either side of the current setting is reported. A knob with no
 * crossing anywhere in range is reported as one the answer does not depend on.
 */
export function findFlips(
  situation: Situation,
  values: Record<string, number>,
  sharpness: number,
  steps = 14,
): { flips: Flip[]; holds: string[] } {
  const here = bestIndex(situation, values, sharpness);
  const built = situation.build(values);
  const flips: Flip[] = [];
  const holds: string[] = [];

  for (const knob of situation.knobs) {
    const current = values[knob.key];
    let found: Flip | null = null;
    for (const dir of ["up", "down"] as const) {
      const span = dir === "up" ? knob.max - current : current - knob.min;
      if (span <= 0) continue;
      let prev = current;
      for (let s = 1; s <= steps; s++) {
        const v = dir === "up" ? current + (span * s) / steps : current - (span * s) / steps;
        const b = bestIndex(situation, { ...values, [knob.key]: v }, sharpness);
        if (b !== here) {
          // Bisect between prev and v for a crossing worth quoting.
          let lo = prev;
          let hi = v;
          for (let k = 0; k < 9; k++) {
            const mid = (lo + hi) / 2;
            if (bestIndex(situation, { ...values, [knob.key]: mid }, sharpness) === here) lo = mid;
            else hi = mid;
          }
          const at = Math.round(hi / knob.step) * knob.step;
          const candidate: Flip = {
            knob: knob.label,
            direction: dir === "up" ? "above" : "below",
            at: knob.format(at),
            atValue: at,
            becomes: built.yourMoves[b].label,
          };
          // Two crossings can exist, one either side. Quote the nearer one:
          // it is the one a practitioner is at risk of walking into.
          if (found === null || Math.abs(at - current) < Math.abs(found.atValue - current)) {
            found = candidate;
          }
          break;
        }
        prev = v;
      }
    }
    if (found) flips.push(found);
    else holds.push(knob.label);
  }
  return { flips, holds };
}

/**
 * The same sweep, over the rival-sharpness control rather than a knob.
 *
 * Swept outward from where the control is now, so the direction reported is
 * the direction the practitioner would have to move it — "below 4" and
 * "above 4" are different warnings and the wording depends on getting it right.
 */
export function sharpnessFlip(
  situation: Situation,
  values: Record<string, number>,
  sharpness: number,
  lo: number,
  hi: number,
  steps = 14,
): { flipsAt: number | null; direction: "above" | "below" | null; becomes: string | null } {
  const built = situation.build(values);
  const here = bestIndex(situation, values, sharpness);
  for (let s = 1; s <= steps; s++) {
    const up = sharpness * Math.pow(hi / sharpness, s / steps);
    const bu = bestIndex(situation, values, up);
    if (bu !== here) return { flipsAt: up, direction: "above", becomes: built.yourMoves[bu].label };
    const down = sharpness * Math.pow(lo / sharpness, s / steps);
    const bd = bestIndex(situation, values, down);
    if (bd !== here) return { flipsAt: down, direction: "below", becomes: built.yourMoves[bd].label };
  }
  return { flipsAt: null, direction: null, becomes: null };
}

// ---------------------------------------------------------------------------
// Repeated play: five policies, one rival process, the same shocks.
// ---------------------------------------------------------------------------

export type PolicyId = "sage" | "bestResponseLast" | "costPlus" | "match" | "nash";

export interface PolicySpec {
  id: PolicyId;
  name: string;
  /** What it does, in one line a practitioner would recognise. */
  rule: string;
  colour: string;
  /** Colour is never the only signal: each line also has its own dash. */
  dash: string;
  /** Two letters, for the end of the line on the chart. */
  short: string;
}

export const POLICIES: PolicySpec[] = [
  {
    id: "sage",
    dash: "none",
    short: "SAGE",
    name: "This solver",
    rule: "Best answer to the whole distribution of what the rival might do, re-solved every round.",
    colour: "var(--accent)",
  },
  {
    id: "bestResponseLast",
    dash: "7 3",
    short: "LAST",
    name: "React to their last move",
    rule: "Assume they repeat last round, then do the best thing against exactly that.",
    colour: "var(--q-driven-text)",
  },
  {
    id: "costPlus",
    dash: "2 3",
    short: "COST",
    name: "Cost-plus",
    rule: "Your own cost plus a fixed markup, snapped to the nearest rung. Ignores the rival entirely.",
    colour: "var(--q-stalled-text)",
  },
  {
    id: "match",
    dash: "10 3 2 3",
    short: "MATCH",
    name: "Match the competitor",
    rule: "Whatever they did last round, do the same.",
    colour: "var(--q-whirlpool-text)",
  },
  {
    id: "nash",
    dash: "1 4",
    short: "NASH",
    name: "Always-Nash",
    rule: "Play the best answer to a rival assumed to be perfectly, relentlessly optimal.",
    colour: "var(--q-none-text)",
  },
];

/** Deterministic PRNG so a run is reproducible from its seed. */
export function rng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 4294967296;
  };
}

function drawFrom(mix: number[], u: number): number {
  let acc = 0;
  for (let j = 0; j < mix.length; j++) {
    acc += mix[j];
    if (u <= acc) return j;
  }
  return mix.length - 1;
}

export interface RoundState {
  /** Per policy: the action it played this round. */
  action: Record<PolicyId, number>;
  /** Per policy: the rival action it faced this round. */
  rivalAction: Record<PolicyId, number>;
  /** Per policy: cumulative value through this round. */
  cumulative: Record<PolicyId, number>;
  round: number;
}

export interface ArenaConfig {
  /** How sharply the simulated rival chases its own best answer. */
  rivalSharpness: number;
  /** 0 = they bet on you repeating last round; 1 = they treat you as unreadable. */
  rivalHedge: number;
  /** Noise added to the rival's choice, as a share of rounds played at random. */
  rivalNoise: number;
  /** Cost-plus markup over your own unit cost, as a fraction. */
  costPlusMarkup: number;
}

function bestReply(u1: Matrix, mix: number[]): number {
  const ev = expectedAgainst(u1, mix);
  let bi = 0;
  for (let i = 1; i < ev.length; i++) if (ev[i] > ev[bi]) bi = i;
  return bi;
}

/**
 * Cost-plus: your own cost times one plus a markup, snapped to the nearest
 * rung. It is a real rule doing real arithmetic, not a rung chosen to lose —
 * which matters, because on some settings it wins.
 */
export function costPlusMove(built: BuiltSituation, markup: number): number {
  const { levels, ownCost } = built;
  if (!levels || ownCost === null) return Math.floor(built.u1.length / 2);
  const target = ownCost * (1 + markup);
  let best = 0;
  for (let i = 1; i < levels.length; i++) {
    if (Math.abs(levels[i] - target) < Math.abs(levels[best] - target)) best = i;
  }
  return best;
}

function pointMass(n: number, i: number): number[] {
  const p = new Array(n).fill(0);
  p[i] = 1;
  return p;
}

/**
 * One round for every policy, each against its own copy of the same rival
 * process driven by the same random stream. Policies never see each other.
 */
export function playRound(
  built: BuiltSituation,
  cfg: ArenaConfig,
  sharpness: number,
  prev: RoundState | null,
  draw: () => number,
): RoundState {
  const nMine = built.u1.length;
  const nTheirs = built.theirMoves.length;
  const opts = { tol: 1e-9, maxIter: 1200 };
  const qre = solveQRE({ u1: built.u1, u2: built.u2 }, lamFor(built, sharpness), opts);
  const nash = solveQRE({ u1: built.u1, u2: built.u2 }, lamFor(built, NASH_SHARPNESS), opts);

  const action = {} as Record<PolicyId, number>;
  const rivalAction = {} as Record<PolicyId, number>;
  const cumulative = {} as Record<PolicyId, number>;

  // One shared draw per round per policy: same seed, same stream position, so
  // a difference between two policies is the policy, not the luck.
  const noiseDraw = draw();
  const pickDraw = draw();

  for (const p of POLICIES) {
    const lastMine = prev ? prev.action[p.id] : Math.floor(nMine / 2);
    const lastTheirs = prev ? prev.rivalAction[p.id] : Math.floor(nTheirs / 2);

    let a: number;
    if (p.id === "sage") a = bestReply(built.u1, qre.sigma2);
    else if (p.id === "bestResponseLast") a = bestReply(built.u1, pointMass(nTheirs, lastTheirs));
    else if (p.id === "costPlus") a = costPlusMove(built, cfg.costPlusMarkup);
    else if (p.id === "match") a = Math.min(nMine - 1, lastTheirs);
    else a = bestReply(built.u1, nash.sigma2);
    action[p.id] = a;

    // The rival expects you to repeat last round, hedged toward "could be
    // anything" by `rivalHedge`; it then chases its own best answer to that
    // belief with its own sharpness, and plays at random on `rivalNoise` of
    // rounds. Nothing about your policy is visible to it beyond your moves.
    const belief = new Array(nMine).fill(cfg.rivalHedge / nMine);
    belief[lastMine] += 1 - cfg.rivalHedge;
    const evThem = built.u2[0].map((_, j) =>
      built.u2.reduce((acc, row, i) => acc + row[j] * belief[i], 0),
    );
    const lamThem = cfg.rivalSharpness / payoffSpread(built);
    const m = Math.max(...evThem.map((v) => lamThem * v));
    const w = evThem.map((v) => Math.exp(lamThem * v - m));
    const z = w.reduce((x, y) => x + y, 0);
    const mix = w.map((v) => v / z);
    const r = noiseDraw < cfg.rivalNoise ? Math.floor(pickDraw * nTheirs) : drawFrom(mix, pickDraw);
    rivalAction[p.id] = Math.min(nTheirs - 1, r);

    cumulative[p.id] = (prev ? prev.cumulative[p.id] : 0) + built.u1[a][rivalAction[p.id]];
  }

  return { action, rivalAction, cumulative, round: (prev?.round ?? 0) + 1 };
}
