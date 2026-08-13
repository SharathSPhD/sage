/* What each explainer changes about a decision.
 *
 * The explainer prose itself is single-sourced from content/theory and is
 * never authored here (see lib/theory.ts). These are the practitioner endings:
 * one sentence of "so do this differently", plus where on the site to do it.
 * They are app copy, not theory, which is why they live in the app.
 */

export interface Takeaway {
  /** The change in behaviour, in the imperative. */
  soWhat: string;
  href: string;
  hrefLabel: string;
}

export const TAKEAWAYS: Record<string, Takeaway> = {
  "01-softmax-and-lambda": {
    soWhat:
      "Stop asking whether your rival is rational and start asking how sharply they chase their best option. It is one number, you can estimate it from their past moves, and it is the difference between planning against one price and planning against four.",
    href: "/solve",
    hrefLabel: "Move that control and watch the answer",
  },
  "02-the-fixed-point": {
    soWhat:
      "Your best move depends on theirs, and theirs on yours. Any planning process that fixes their behaviour first and optimises against it is answering a different question than the one you have.",
    href: "/play",
    hrefLabel: "Watch that assumption cost money over a hundred rounds",
  },
  "03-qre-vs-mixed-nash": {
    soWhat:
      "Assuming the other side is perfect is not the conservative choice — it is a specific, usually wrong, forecast. Price the difference before you adopt it.",
    href: "/solve",
    hrefLabel: "Compare the two recommendations side by side",
  },
  "04-maxent": {
    soWhat:
      "When you genuinely do not know what they will do, the honest forecast is the most spread-out one consistent with what you do know. Anything sharper is you adding information you do not have.",
    href: "/situations/allocation",
    hrefLabel: "See a decision where spreading out is the answer",
  },
  "05-gibbs-and-potential-games": {
    soWhat:
      "Some situations have a hill everyone is climbing; in those, a change you make moves things where you expect and comparative statics can be trusted. Find out which kind you are in before trusting a sensitivity table.",
    href: "/situations/routing",
    hrefLabel: "A real system that is exactly this kind",
  },
  "06-detailed-balance-and-currents": {
    soWhat:
      "If your market cycles rather than settling, waiting for it to stabilise is not a strategy — it will not. Time your moves against the cycle instead of trying to end it.",
    href: "/diagnose",
    hrefLabel: "Find out whether yours cycles",
  },
  "07-reciprocity": {
    soWhat:
      "Measure how much your price moves theirs and how much theirs moves yours. When the two are different sizes, one of you structurally leads — and that asymmetry, not the average elasticity, is the thing to act on.",
    href: "/diagnose",
    hrefLabel: "Measure it on your own series",
  },
  "08-elasticity-vs-lambda": {
    soWhat:
      "A demand elasticity and a rival's decisiveness look the same in a regression and mean opposite things for what you should do. Separate them before you set a price on either.",
    href: "/situations/pricing",
    hrefLabel: "Change one, then the other, and compare",
  },
  "09-the-one-price-objection": {
    soWhat:
      "If your rival really does set one price and never move, none of this buys you anything and you should say so. Check that first; it takes a minute and it is the cheapest way to avoid a wrong model.",
    href: "/diagnose",
    hrefLabel: "Run the check",
  },
  "10-the-same-machinery-everywhere": {
    soWhat:
      "The pricing answer, the bidding answer and the routing answer come out of one piece of arithmetic. If you have solved one of these decisions well, you already know how to set up the others.",
    href: "/situations",
    hrefLabel: "The same machinery on five decisions",
  },
};
