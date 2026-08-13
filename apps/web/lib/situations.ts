/* The situations the app can solve, in the words their practitioners use.
 *
 * Each situation turns a handful of numbers a practitioner already knows —
 * a unit cost, a bid ladder, a budget — into the two payoff tables the solver
 * needs. Nothing here invents data: every default is either an engine-exact
 * construction (allocation, standards) or a range taken from the dataset named
 * in `sourceNote`, and every page states which.
 *
 * The payoff builders are deliberately plain arithmetic so the whole thing runs
 * in the browser between keystrokes. The committed answer for the same numbers
 * comes back from the deployed float64 solver via /api.
 */

export interface Knob {
  key: string;
  /** What a practitioner would call it. No symbols. */
  label: string;
  /** One line, only if the label is not self-evident. */
  help?: string;
  min: number;
  max: number;
  step: number;
  /** Rendered value, e.g. "$0.95" or "3 weeks". */
  format: (v: number) => string;
  /** Whose number this is — changes the sensitivity wording. */
  side: "you" | "rival" | "market";
}

export interface Move {
  /** What you would say in the meeting: "$1.29". */
  label: string;
  /** Optional qualifier under the label: "hold", "match them". */
  note?: string;
}

export interface BuiltSituation {
  /** Your payoff. u[i][j] = you play i, they play j. */
  u1: number[][];
  /** Their payoff, same indexing. */
  u2: number[][];
  yourMoves: Move[];
  theirMoves: Move[];
  /** Sentence completing "The recommendation is to …". */
  decide: (m: Move) => string;
  /** Money/units label for the value, e.g. "per store-week". */
  valueUnit: string;
  formatValue: (v: number) => string;
  /**
   * The numeric level behind each of your moves — the price, the bid — and
   * your own unit cost in the same units. Only the repeated-play arena needs
   * these: cost-plus is a markup over cost, so it has to be able to do the
   * arithmetic rather than be handed a rung number. Null where a situation
   * has no such scale (picking a standard is not a price).
   */
  levels: number[] | null;
  ownCost: number | null;
}

export interface Situation {
  id: string;
  /** The name a practitioner uses. */
  name: string;
  /** The decision, in one line, second person. */
  decision: string;
  /** Two sentences of setting. No theory. */
  setting: string;
  /** Who the other side is. */
  rival: string;
  /** Where the numbers come from. Shown verbatim on the page. */
  sourceNote: string;
  /** true when the numbers are a plausible stand-in rather than measured. */
  illustrative: boolean;
  /** Label for the rival-sharpness control on this situation. */
  sharpnessLabel: string;
  /**
   * Rival sharpness, in units of the rival's own payoff spread, so the same
   * number means the same thing in dollars, share points or contract margin.
   * Converted to the solver's raw setting by `lamFor` in lib/solve.ts.
   */
  defaults: Record<string, number>;
  defaultSharpness: number;
  knobs: Knob[];
  build: (v: Record<string, number>) => BuiltSituation;
  /** The thing worth learning here, phrased as an experiment to run. */
  tryThis: { title: string; body: string }[];
  href: string;
}

const usd = (v: number) => `$${v.toFixed(2)}`;
const usd0 = (v: number) => `$${Math.round(v).toLocaleString()}`;
const k$ = (v: number) => `$${Math.round(v / 1000)}k`;

// ---------------------------------------------------------------------------
// 1 · Weekly shelf price against one rival
// ---------------------------------------------------------------------------

/** Multinomial-logit share: you, them, and walking out with neither. */
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
  name: "Weekly shelf price",
  decision: "What do I price at on Monday, with one rival moving the same week?",
  setting:
    "You and one rival set a shelf price for the same category every week. You both see last week's prices; neither of you sees this week's until it is set.",
  rival: "the other brand on the shelf",
  sourceNote:
    "Illustrative. The price ladder ($1.09–$1.89 in 10¢ steps) and the wholesale margin are the range of the Dominick's canned-soup panel this project uses elsewhere; the demand model is a standard logit share model and is not fitted to that panel. Put your own numbers in.",
  illustrative: true,
  sharpnessLabel: "How closely the rival tracks their best price",
  defaultSharpness: 8,
  defaults: {
    yourCost: 1.0,
    theirCost: 1.0,
    volume: 400,
    sensitivity: 3.6,
    outside: 1.65,
  },
  knobs: [
    { key: "yourCost", label: "Your unit cost", min: 0.6, max: 1.35, step: 0.01, format: usd, side: "you" },
    { key: "theirCost", label: "What you think their unit cost is", min: 0.6, max: 1.35, step: 0.01, format: usd, side: "rival" },
    { key: "volume", label: "Category units per store-week", min: 100, max: 1200, step: 20, format: (v) => `${Math.round(v)} units`, side: "market" },
    {
      key: "sensitivity",
      label: "Price sensitivity",
      help: "Higher means shoppers switch harder for the same 10¢.",
      min: 1.0,
      max: 6.0,
      step: 0.1,
      format: (v) => v.toFixed(1),
      side: "market",
    },
    {
      key: "outside",
      label: "Price of the next best thing on the shelf",
      help: "Private label, a different format, or leaving without one.",
      min: 1.2,
      max: 2.4,
      step: 0.05,
      format: usd,
      side: "market",
    },
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
      decide: (m) => `price at ${m.label} this week`,
      valueUnit: "gross margin per store-week",
      formatValue: usd0,
      levels: prices,
      ownCost: v.yourCost,
    };
  },
  tryThis: [
    {
      title: "Your price does not track your cost",
      body: "Drag your unit cost down. Nothing happens for the first several cents — then the price steps down a whole rung at once, and a smaller cost saving has bought a larger price cut. Your shelf price tracks the rival's price and the step size of your own ladder, not your cost. Cost-plus gets this backwards in both directions.",
    },
    {
      title: "Make shoppers pickier",
      body: "Push price sensitivity up by a point. The recommendation drops a rung and the spread of outcomes widens: when shoppers switch harder for the same 10¢, your result depends more on what the rival does than on what you do — and the worst case gets worse faster than the average does.",
    },
    {
      title: "Turn the rival into a machine",
      body: "Slide their tracking to the top. Their price collapses onto one rung and your spread of outcomes narrows, because there is almost nothing left to hedge against. Watch what that does to the gap between the top two options: a predictable rival makes the decision easier and the margin thinner at the same time.",
    },
  ],
  href: "/situations/pricing",
};

// ---------------------------------------------------------------------------
// 2 · Sealed bid for a supply contract
// ---------------------------------------------------------------------------

const PROCUREMENT: Situation = {
  id: "procurement",
  name: "Sealed bid for a contract",
  decision: "What do I bid, knowing one credible rival is bidding too?",
  setting:
    "A buyer runs a sealed tender for a year of supply. Lowest compliant bid wins the whole thing; a tie splits it. You get one shot and you do not see their number.",
  rival: "the incumbent supplier",
  sourceNote:
    "Illustrative. A single-round, lowest-price-wins tender on a $100k contract, with a bid ladder in $4k steps. Costs and the reserve are yours to set — nothing here is drawn from a real tender.",
  illustrative: true,
  sharpnessLabel: "How disciplined the rival's bidding is",
  defaultSharpness: 8,
  defaults: {
    yourCost: 85000,
    theirCost: 88000,
    reserve: 112000,
    preference: 0,
  },
  knobs: [
    { key: "yourCost", label: "Your cost to deliver", min: 70000, max: 105000, step: 1000, format: k$, side: "you" },
    { key: "theirCost", label: "What you think it costs them", min: 70000, max: 105000, step: 1000, format: k$, side: "rival" },
    { key: "reserve", label: "Buyer's walk-away price", min: 92000, max: 125000, step: 1000, format: k$, side: "market" },
    {
      key: "preference",
      label: "How much more the buyer will pay for you",
      help: "Switching cost, incumbency, a preferred-supplier score — worth this much in bid terms.",
      min: 0,
      max: 12000,
      step: 500,
      format: k$,
      side: "market",
    },
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
        // The buyer scores your bid as `mine - preference`.
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
      decide: (m) => `bid ${m.label}`,
      valueUnit: "expected contribution on this contract",
      formatValue: usd0,
      levels: bids,
      ownCost: v.yourCost,
    };
  },
  tryThis: [
    {
      title: "Your cost barely matters. Theirs decides everything",
      body: "Move your own cost $10k either way: the bid holds. Move your estimate of their cost $10k down: the recommendation drops two rungs. The number worth spending a week on is theirs, not yours.",
    },
    {
      title: "Price the relationship",
      body: "Set the buyer's preference for you to $6k. The recommended bid rises by roughly that much and the expected contribution rises more — a preference you never invoice for is money you can bid with.",
    },
    {
      title: "Watch the reserve bite",
      body: "Drop the walk-away price to $96k. Everything above it becomes worthless, the ladder collapses, and the bid that was safely profitable is now the one that loses you the contract.",
    },
  ],
  href: "/situations/procurement",
};

// ---------------------------------------------------------------------------
// 3 · Which standard to back
// ---------------------------------------------------------------------------

const STANDARDS: Situation = {
  id: "standards",
  name: "Which standard to back",
  decision: "Do I adopt their format, hold mine, or go it alone?",
  setting:
    "Two suppliers each pick a format — a connector, a file spec, a data schema. Customers buy more when both of you are on the same one, and the side that switches pays for it.",
  rival: "the other supplier in the category",
  sourceNote:
    "Illustrative and deliberately schematic: market growth from agreement, an installed-base advantage, and a one-off switching cost, all in the same units. It is a shape, not a forecast — the point is which way the answer tips, not the size of the number.",
  illustrative: true,
  sharpnessLabel: "How decisively the rival commits",
  defaultSharpness: 8,
  defaults: {
    agreement: 10,
    yourBase: 7,
    theirBase: 6,
    switchCost: 3,
    goneAlone: 2,
  },
  knobs: [
    { key: "agreement", label: "Extra market if you both land on one format", min: 0, max: 20, step: 0.5, format: (v) => v.toFixed(1), side: "market" },
    { key: "yourBase", label: "Your installed base on your format", min: 0, max: 14, step: 0.5, format: (v) => v.toFixed(1), side: "you" },
    { key: "theirBase", label: "Their installed base on theirs", min: 0, max: 14, step: 0.5, format: (v) => v.toFixed(1), side: "rival" },
    { key: "switchCost", label: "Cost of moving off your own format", min: 0, max: 12, step: 0.5, format: (v) => v.toFixed(1), side: "you" },
    { key: "goneAlone", label: "What a private format is worth on its own", min: 0, max: 10, step: 0.5, format: (v) => v.toFixed(1), side: "you" },
  ],
  build: (v) => {
    // 0 = back your format, 1 = adopt theirs, 2 = go proprietary.
    const you = ["Back your format", "Adopt theirs", "Go proprietary"];
    const them = ["Back their format", "Adopt yours", "Go proprietary"];
    const u1: number[][] = [];
    const u2: number[][] = [];
    for (let i = 0; i < 3; i++) {
      u1.push([]);
      u2.push([]);
      for (let j = 0; j < 3; j++) {
        // Agreement happens when both name the same format.
        const bothOnYours = i === 0 && j === 1;
        const bothOnTheirs = i === 1 && j === 0;
        let a = 0;
        let b = 0;
        if (bothOnYours) {
          a = v.yourBase + v.agreement;
          b = v.agreement - v.switchCost;
        } else if (bothOnTheirs) {
          a = v.agreement - v.switchCost;
          b = v.theirBase + v.agreement;
        } else {
          a = i === 2 ? v.goneAlone : i === 0 ? v.yourBase : -v.switchCost;
          b = j === 2 ? v.goneAlone : j === 0 ? v.theirBase : -v.switchCost;
        }
        u1[i].push(a);
        u2[i].push(b);
      }
    }
    return {
      u1,
      u2,
      yourMoves: you.map((label) => ({ label })),
      theirMoves: them.map((label) => ({ label })),
      decide: (m) => m.label.toLowerCase(),
      valueUnit: "share points of the category, per year",
      formatValue: (x) => `${x >= 0 ? "" : "−"}${Math.abs(x).toFixed(1)} pts`,
      levels: null,
      ownCost: null,
    };
  },
  tryThis: [
    {
      title: "Being right about the format does not win it",
      body: "Give yourself the bigger installed base and the recommendation is to hold. Now give it to them by two points: the answer flips to adopting theirs even though nothing about the formats changed. Standards go to the larger base, not the better spec.",
    },
    {
      title: "Proprietary is not the safe option",
      body: "Raise your switching cost until going proprietary wins. Look at what it is worth: less than either agreement. Going alone is the least-bad option, never a good one — and it stops being least-bad the moment the rival's base grows.",
    },
    {
      title: "Deadlock is a real answer",
      body: "Set both installed bases equal and both switching costs high. The rival's likely moves stop concentrating: nobody moves first, and the expected value of every option collapses toward the same number. That flat spread is the tell that the decision needs a side deal, not a better analysis.",
    },
  ],
  href: "/situations/standards",
};

export const SITUATIONS: Situation[] = [PRICING, PROCUREMENT, STANDARDS];

export function getSituation(id: string): Situation {
  return SITUATIONS.find((s) => s.id === id) ?? PRICING;
}

/** The two situations solved server-side by the engine, listed in the gallery. */
export const ENGINE_SITUATIONS = [
  {
    id: "routing",
    name: "Where to put the toll",
    decision: "I can price or close one link. Which one, and does it actually help?",
    setting:
      "A whole city's traffic re-arranges itself after any change you make. Pick a road, charge for it, and see where the queue moves to.",
    sourceNote:
      "Real data. The Sioux Falls benchmark network — 24 junctions, 76 links, its published demand table — solved by the deployed route-choice solver on every change.",
    illustrative: false,
    href: "/situations/routing",
  },
  {
    id: "allocation",
    name: "Splitting a fixed budget",
    decision: "Three accounts, one budget, a rival splitting theirs. Where do I put it?",
    setting:
      "You and a competitor each divide a fixed resource — heads, spend, engineering weeks — across the same three accounts. Whoever commits more takes the account.",
    sourceNote:
      "Engine-exact. Every allocation is enumerated and solved exactly by the deployed solver; there is no sampling and no fitted parameter.",
    illustrative: false,
    href: "/situations/allocation",
  },
] as const;
