import type { Metadata } from "next";
import Link from "next/link";
import { PricingSolver } from "../../components/solvers/PricingSolver";

export const metadata: Metadata = {
  title: "Pricing",
  description: "Set a price against a rival setting one too: price, profit, margin and elasticities.",
};

export default function PricingPage() {
  return (
    <div className="wrap situation-page">
      <p className="crumb">
        <Link href="/situations">All problems</Link>
      </p>
      <h1 className="surface-title">Pricing</h1>
      <p className="surface-lede">
        Two firms set a price on the same grid in the same period, each seeing the other&apos;s last move and not this
        one. Returns the price to set, the profit at it, and how both change with your inputs.
      </p>
      <PricingSolver />
      <section className="card next-step">
        <h2>In Python</h2>
        <p>
          <code>
            sq.PricingProblem(costs=[1.00, 1.05], grid=(1.09, 1.89, 0.10), demand=sq.LogitDemand(3.6, [5.94, 5.94],
            market_size=400), precision=1.5).solve()
          </code>{" "}
          returns the solution this page is rendering at its defaults — the quality terms are the outside option price
          times the price coefficient. See <a href="https://sharathsphd.github.io/sage/">the docs</a> for{" "}
          <code>LinearDemand</code> and <code>CustomDemand</code>.
        </p>
      </section>
    </div>
  );
}
