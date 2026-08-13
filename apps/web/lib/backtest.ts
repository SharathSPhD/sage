/* Repeated-round backtest: five decision rules, one simulated rival, one seed.
 *
 * Each rule plays its own copy of the situation against its own copy of the
 * same rival process, driven from a single seeded stream, so a gap between two
 * lines is the rule and not the draw. Everything here runs in the browser
 * because a hundred rounds times five rules is five hundred solves; the
 * single-round answer on the problem pages comes from /v1/solve/* instead.
 */

import { solveQRE, type Matrix } from "./qre";
import type { BuiltSituation } from "./situations";

/** Precision, in payoff-spread units, at which a side is Nash in all but name. */
const NASH_SHARPNESS = 600;

/** The rival's payoff spread — the scale that makes precision mean one thing. */
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

/** On-screen precision is in payoff-spread units; the solver wants it raw. */
export function lamFor(built: BuiltSituation, sharpness: number): number {
  return sharpness / payoffSpread(built);
}

function expectedAgainst(u1: Matrix, mix: number[]): number[] {
  return u1.map((row) => row.reduce((a, v, j) => a + v * mix[j], 0));
}

export type PolicyId = "sage" | "bestResponseLast" | "costPlus" | "match" | "nash";

export interface PolicySpec {
  id: PolicyId;
  name: string;
  /** What it does, in one line. */
  rule: string;
  colour: string;
  /** Colour is never the only signal: each line also has its own dash. */
  dash: string;
  /** A short tag for the end of the line on the chart. */
  short: string;
}

export const POLICIES: PolicySpec[] = [
  {
    id: "sage",
    dash: "none",
    short: "QRE",
    name: "Solver",
    rule: "Best reply to the rival's whole move distribution, re-solved every round.",
    colour: "var(--accent)",
  },
  {
    id: "bestResponseLast",
    dash: "7 3",
    short: "LAST",
    name: "Best reply to last",
    rule: "Assume they repeat last round, then play the best reply to exactly that.",
    colour: "var(--q-driven-text)",
  },
  {
    id: "costPlus",
    dash: "2 3",
    short: "COST",
    name: "Cost-plus",
    rule: "Own cost plus a fixed markup, snapped to the nearest level. Ignores the rival.",
    colour: "var(--q-stalled-text)",
  },
  {
    id: "match",
    dash: "10 3 2 3",
    short: "MATCH",
    name: "Match the rival",
    rule: "Play whatever they played last round.",
    colour: "var(--q-whirlpool-text)",
  },
  {
    id: "nash",
    dash: "1 4",
    short: "NASH",
    name: "Always-Nash",
    rule: "Best reply to a rival assumed to optimise exactly, every round.",
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
  action: Record<PolicyId, number>;
  rivalAction: Record<PolicyId, number>;
  cumulative: Record<PolicyId, number>;
  round: number;
}

export interface ArenaConfig {
  /** How sharply the simulated rival chases its own best reply. */
  rivalSharpness: number;
  /** 0 = they bet on you repeating last round; 1 = they treat you as unreadable. */
  rivalHedge: number;
  /** Share of rounds the rival plays at random. */
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

/** Cost-plus: own cost times one plus a markup, snapped to the nearest level. */
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

/** One round for every rule, each against its own copy of the same rival. */
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

  // One shared draw per round per rule: same seed, same stream position, so a
  // difference between two rules is the rule, not the luck.
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
    // anything" by rivalHedge; it then chases its own best reply to that belief
    // with its own precision, and plays at random on rivalNoise of rounds.
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
