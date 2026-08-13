import type { Metadata } from "next";
import Link from "next/link";
import { AuctionSolver } from "../../components/solvers/AuctionSolver";

export const metadata: Metadata = {
  title: "Auction and tender — SAGE",
  description: "Sealed-bid tender or sale: the bid to submit, expected surplus and win probability.",
};

export default function AuctionPage() {
  return (
    <div className="wrap situation-page">
      <p className="crumb">
        <Link href="/situations">All problems</Link>
      </p>
      <h1 className="surface-title">Auction and tender</h1>
      <p className="surface-lede">
        One sealed round against one credible rival. In a tender the lowest eligible bid wins and the reserve is the
        buyer&apos;s ceiling; in a sale the highest wins and the reserve is a floor.
      </p>
      <AuctionSolver />
      <section className="card next-step">
        <h2>In Python</h2>
        <p>
          <code>sq.AuctionProblem(costs=[85000, 88000], grid=(88000, 116000, 4000), reserve=112000, precision=5e-4).solve()</code>{" "}
          returns the same solution. Pass <code>values=</code> instead of <code>costs=</code> for a sale.
        </p>
      </section>
    </div>
  );
}
