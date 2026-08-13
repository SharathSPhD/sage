import type { Metadata } from "next";
import Link from "next/link";
import { Workbench } from "../components/solvers/Workbench";

export const metadata: Metadata = {
  title: "Solve",
  description:
    "Pricing, auction, electricity, routing, allocation and coordination problems: enter your numbers, get the price, bid, offer, toll or split, and see what moves it.",
};

export default function SolvePage() {
  return (
    <div className="wrap page">
      <h1 className="surface-title">Studio</h1>
      <p className="surface-lede">
        Six problem types over one solver. Every number below is a field of the solution returned by{" "}
        <code>/v1/solve</code>; the same call from Python returns the same answer. Drag anything and the whole answer
        moves with it.
      </p>
      <Workbench />
      <section className="card next-step">
        <h2>Also here</h2>
        <p>
          Each problem type has its own page under <Link href="/situations">Problems</Link>, with more room for its
          visual. To solve on your own numbers, load a file in <Link href="/data">bring your own data</Link>. Over
          repeated rounds, the <Link href="/play">backtest</Link> compares this solver&apos;s move with the rules teams
          actually use.
        </p>
      </section>
    </div>
  );
}
