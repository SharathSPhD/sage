import type { Metadata } from "next";
import { Arena } from "./Arena";

export const metadata: Metadata = {
  title: "Backtest",
  description:
    "How five decision rules perform over repeated rounds against the same simulated rival on the same draws.",
};

export default function PlayPage() {
  return (
    <div className="wrap" style={{ paddingTop: "2.4rem" }}>
      <h1 className="surface-title">Backtest</h1>
      <p className="surface-lede">
        How a strategy performs over time against common rules. Five rules face the same simulated rival, from the same
        starting position, on the same seeded draws, so the gap at the end is the rule and not the luck. Change the
        rival or your costs mid-run to see which rules survive the change.
      </p>
      <Arena />
    </div>
  );
}
