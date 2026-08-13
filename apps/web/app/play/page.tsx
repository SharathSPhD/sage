import type { Metadata } from "next";
import { Arena } from "./Arena";

export const metadata: Metadata = {
  title: "Run it against the usual rules — SAGE",
  description:
    "Five ways of deciding, the same rival, the same luck, over a hundred rounds. Cost-plus, matching, reacting to their last move, always-Nash, and this solver.",
};

export default function PlayPage() {
  return (
    <div className="wrap" style={{ paddingTop: "2.4rem" }}>
      <h1 className="surface-title">Run it against the usual rules</h1>
      <p className="surface-lede">
        One answer is easy to argue with. A hundred rounds is not. Every rule below faces the same rival, on the same
        random draws, from the same starting position — so the gap at the end is the rule, not the luck. Change the
        world halfway through and watch which ones survive it. Roughly one setting in seven leaves a simpler rule ahead,
        and there is a button below that takes you straight to one.
      </p>
      <Arena />
    </div>
  );
}
