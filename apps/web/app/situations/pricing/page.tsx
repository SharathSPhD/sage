import type { Metadata } from "next";
import Link from "next/link";
import { SolveStudio } from "../../solve/SolveStudio";
import { getSituation } from "../../../lib/situations";

const S = getSituation("pricing");

export const metadata: Metadata = {
  title: "Weekly shelf price — SAGE",
  description: S.decision,
};

export default function PricingSituation() {
  return (
    <div className="wrap situation-page">
      <p className="crumb">
        <Link href="/situations">All situations</Link>
      </p>
      <h1 className="surface-title">{S.name}</h1>
      <p className="surface-lede">{S.setting}</p>
      <SolveStudio fixedSituation="pricing" />
      <section className="try-this" aria-labelledby="try-heading">
        <h2 id="try-heading">Three things worth trying</h2>
        <div className="try-grid">
          {S.tryThis.map((t) => (
            <article key={t.title} className="card">
              <h3>{t.title}</h3>
              <p>{t.body}</p>
            </article>
          ))}
        </div>
      </section>
      <section className="card next-step">
        <h2>Then what</h2>
        <p>
          A single week&apos;s answer is not the whole job. <Link href="/play">Run it for fifty weeks</Link> against
          cost-plus, matching, and reacting to their last move, and see which one is actually ahead by the end.
        </p>
      </section>
    </div>
  );
}
