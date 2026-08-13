import type { Metadata } from "next";
import Link from "next/link";
import { ElectricitySolver } from "../../components/solvers/ElectricitySolver";

export const metadata: Metadata = {
  title: "Electricity offers — SAGE",
  description: "Offer a block into a uniform-price market: offer price, clearing price, revenue and dispatch.",
};

export default function ElectricityPage() {
  return (
    <div className="wrap situation-page">
      <p className="crumb">
        <Link href="/situations">All problems</Link>
      </p>
      <h1 className="surface-title">Electricity offers</h1>
      <p className="surface-lede">
        Two generators offer capacity into a uniform-price auction. The market takes the cheapest offers until demand
        is met and pays everyone dispatched the marginal price. Returns the offer to submit, where the market clears,
        and what you are paid.
      </p>
      <ElectricitySolver />
      <section className="card next-step">
        <h2>In Python</h2>
        <p>
          <code>
            sq.ElectricityProblem(costs=[20.0, 22.0], offers=(20.0, 60.0, 5.0), capacities=[100.0, 100.0], demand=80.0,
            precision=0.05).solve()
          </code>{" "}
          returns the same solution. For measured day-ahead and real-time series, see{" "}
          <Link href="/markets">the market reading</Link>.
        </p>
      </section>
    </div>
  );
}
