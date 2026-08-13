import type { Metadata } from "next";
import Link from "next/link";
import { MatrixSolver } from "../../components/solvers/MatrixSolver";

export const metadata: Metadata = {
  title: "Payoff table — SAGE",
  description: "A three-by-three payoff table of your own, solved for both sides' move distributions.",
};

export default function StandardsPage() {
  return (
    <div className="wrap situation-page">
      <p className="crumb">
        <Link href="/situations">All problems</Link>
      </p>
      <h1 className="surface-title">Payoff table</h1>
      <p className="surface-lede">
        Two suppliers each pick a format — a connector, a file spec, a schema. Agreement grows the market, the side
        that switches pays for it. The payoffs are built from the five numbers below and solved as a table.
      </p>
      <MatrixSolver />
      <section className="card next-step">
        <h2>An arbitrary table</h2>
        <p>
          For any payoff tensor rather than this parameterisation, <Link href="/lab">the bench</Link> takes one
          directly, and the Python library takes n-player tensors of any shape.
        </p>
      </section>
    </div>
  );
}
