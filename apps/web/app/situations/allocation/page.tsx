import type { Metadata } from "next";
import Link from "next/link";
import { AllocationSolver } from "../../components/solvers/AllocationSolver";

export const metadata: Metadata = {
  title: "Budget allocation — SAGE",
  description: "Split a fixed budget across contested accounts: the split, win probability and expected value.",
};

export default function AllocationPage() {
  return (
    <div className="wrap situation-page">
      <p className="crumb">
        <Link href="/situations">All problems</Link>
      </p>
      <h1 className="surface-title">Budget allocation</h1>
      <p className="surface-lede">
        You and a rival each divide a fixed budget — heads, spend, engineering weeks — across the same three accounts.
        Whoever commits more takes the account. Every split is enumerated, so the weights below are exact.
      </p>
      <AllocationSolver />
      <section className="card next-step">
        <h2>In Python</h2>
        <p>
          <code>sq.AllocationProblem(budget=5, field_values=[1.0, 1.0, 2.0], precision=2.0).solve()</code> returns the
          same solution. The units are yours: sales heads, artillery, engineering weeks.
        </p>
      </section>
    </div>
  );
}
