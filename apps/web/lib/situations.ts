/* The two settings the repeated-round backtest runs in.
 *
 * A backtest needs a payoff table it can evaluate a hundred times per second
 * in the browser, which is why these are plain arithmetic rather than a call
 * per round. They are the same specifications the pricing and auction
 * endpoints take: a logit price ladder and a lowest-bid-wins tender.
 *
 * Single-round answers do not come from here — those come from /v1/solve/*.
 */

export interface Knob {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
}

export interface Move {
  label: string;
}

export interface BuiltSituation {
  /** Your payoff. u[i][j] = you play i, they play j. */
  u1: number[][];
  /** Their payoff, same indexing. */
  u2: number[][];
  yourMoves: Move[];
  theirMoves: Move[];
  formatValue: (v: number) => string;
  /** The price or bid behind each of your moves, and your own unit cost in the
   * same units — cost-plus needs the arithmetic, not a rung number. */
  levels: number[] | null;
  ownCost: number | null;
}

export interface Situation {
  id: string;
  name: string;
  /** One line naming the model, printed under the chart. */
  model: string;
  defaults: Record<string, number>;
  defaultSharpness: number;
  knobs: Knob[];
  build: (v: Record<string, number>) => BuiltSituation;
}

const usd = (v: number) => `$${v.toFixed(2)}`;
const usd0 = (v: number) => `$${Math.round(v).toLocaleString()}`;
const k$ = (v: number) => `$${Math.round(v / 1000)}k`;

/** Multinomial-logit share: you, them, and the outside option. */
function shares(pYou: number, pThem: number, pOut: number, sensitivity: number) {
  const e = [
    Math.exp(-sensitivity * pYou),
    Math.exp(-sensitivity * pThem),
    Math.exp(-sensitivity * pOut),
  ];
  const z = e[0] + e[1] + e[2];
  return [e[0] / z, e[1] / z];
}

export function priceLadder(low: number, step: number, n: number): number[] {
  return Array.from({ length: n }, (_, i) => low + i * step);
}

const PRICING: Situation = {
  id: "pricing",
  name: "Pricing",
  model: "Logit demand, 2 firms, 9 price levels from $1.09 to $1.89, profit per store-week.",
  defaultSharpness: 8,
  defaults: { yourCost: 1.0, theirCost: 1.0, volume: 400, sensitivity: 3.6, outside: 1.65 },
  knobs: [
    { key: "yourCost", label: "Your unit cost", min: 0.6, max: 1.35, step: 0.01, format: usd },
    { key: "theirCost", label: "Rival unit cost", min: 0.6, max: 1.35, step: 0.01, format: usd },
    { key: "volume", label: "Category units per store-week", min: 100, max: 1200, step: 20, format: (v) => `${Math.round(v)} units` },
    { key: "sensitivity", label: "Price sensitivity", min: 1.0, max: 6.0, step: 0.1, format: (v) => v.toFixed(1) },
    { key: "outside", label: "Outside option price", min: 1.2, max: 2.4, step: 0.05, format: usd },
  ],
  build: (v) => {
    const prices = priceLadder(1.09, 0.1, 9);
    const u1: number[][] = [];
    const u2: number[][] = [];
    for (let i = 0; i < prices.length; i++) {
      u1.push([]);
      u2.push([]);
      for (let j = 0; j < prices.length; j++) {
        const [sYou, sThem] = shares(prices[i], prices[j], v.outside, v.sensitivity);
        u1[i].push((prices[i] - v.yourCost) * v.volume * sYou);
        u2[i].push((prices[j] - v.theirCost) * v.volume * sThem);
      }
    }
    const moves = prices.map((p) => ({ label: usd(p) }));
    return {
      u1,
      u2,
      yourMoves: moves,
      theirMoves: moves.map((m) => ({ ...m })),
      formatValue: usd0,
      levels: prices,
      ownCost: v.yourCost,
    };
  },
};

const PROCUREMENT: Situation = {
  id: "procurement",
  name: "Tender",
  model: "Sealed bid, lowest eligible bid wins, 8 levels in $4k steps, expected contribution per tender.",
  defaultSharpness: 8,
  defaults: { yourCost: 85000, theirCost: 88000, reserve: 112000, preference: 0 },
  knobs: [
    { key: "yourCost", label: "Your cost to deliver", min: 70000, max: 105000, step: 1000, format: k$ },
    { key: "theirCost", label: "Rival cost to deliver", min: 70000, max: 105000, step: 1000, format: k$ },
    { key: "reserve", label: "Buyer's walk-away price", min: 92000, max: 125000, step: 1000, format: k$ },
    { key: "preference", label: "Buyer's preference for you", min: 0, max: 12000, step: 500, format: k$ },
  ],
  build: (v) => {
    const bids = Array.from({ length: 8 }, (_, i) => 88000 + i * 4000);
    const u1: number[][] = [];
    const u2: number[][] = [];
    for (let i = 0; i < bids.length; i++) {
      u1.push([]);
      u2.push([]);
      for (let j = 0; j < bids.length; j++) {
        const mine = bids[i];
        const theirs = bids[j];
        const scoredMine = mine - v.preference;
        const youIn = mine <= v.reserve;
        const themIn = theirs <= v.reserve;
        let pYou = 0;
        if (youIn && (!themIn || scoredMine < theirs)) pYou = 1;
        else if (youIn && themIn && scoredMine === theirs) pYou = 0.5;
        const pThem = themIn ? (youIn ? 1 - pYou : 1) : 0;
        u1[i].push(pYou * (mine - v.yourCost));
        u2[i].push(pThem * (theirs - v.theirCost));
      }
    }
    return {
      u1,
      u2,
      yourMoves: bids.map((b) => ({ label: k$(b) })),
      theirMoves: bids.map((b) => ({ label: k$(b) })),
      formatValue: usd0,
      levels: bids,
      ownCost: v.yourCost,
    };
  },
};

export const SITUATIONS: Situation[] = [PRICING, PROCUREMENT];

export function getSituation(id: string): Situation {
  return SITUATIONS.find((s) => s.id === id) ?? PRICING;
}
