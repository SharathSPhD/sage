import type { Metadata } from "next";
import Link from "next/link";
import { SolveStudio } from "./SolveStudio";

export const metadata: Metadata = {
  title: "Solve a situation — SAGE",
  description:
    "Pick the situation you are in and get the move to make, what the other side is likely to do, what it is worth, and what would change the answer.",
};

export default function SolvePage() {
  return (
    <div className="wrap" style={{ paddingTop: "2.4rem" }}>
      <h1 className="surface-title">Solve a situation</h1>
      <SolveStudio />
      <section className="card next-step">
        <h2>Then what</h2>
        <p>
          One round is one round. <Link href="/play">Run a hundred of them</Link> against cost-plus, matching the
          competitor, reacting to their last move and assuming they are perfect — same rival, same luck — and see which
          rule is actually ahead at the end. Or take{" "}
          <Link href="/situations">a different situation</Link> with the same machinery underneath.
        </p>
      </section>
    </div>
  );
}
