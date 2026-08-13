import type { Metadata } from "next";
import Link from "next/link";
import { Workbench } from "../components/solvers/Workbench";

export const metadata: Metadata = {
  title: "Solve — SAGE",
  description:
    "Pricing, auction, electricity and allocation problems: enter your numbers, get the price, bid, offer or split, and see what moves it.",
};

export default function SolvePage() {
  return (
    <div className="wrap" style={{ paddingTop: "2.4rem" }}>
      <h1 className="surface-title">Solve</h1>
      <p className="surface-lede">
        Four problem types over one solver. Every number below is a field of the solution returned by{" "}
        <code>/v1/solve</code>; the same call from Python returns the same answer.
      </p>
      <Workbench />
      <section className="card next-step">
        <h2>Also here</h2>
        <p>
          <Link href="/situations/routing">Traffic assignment with tolls</Link> on the Sioux Falls network, and{" "}
          <Link href="/situations/standards">a payoff table you write yourself</Link>. Over repeated rounds, the{" "}
          <Link href="/play">backtest</Link> compares this solver&apos;s move with the rules teams actually use.
        </p>
      </section>
    </div>
  );
}
