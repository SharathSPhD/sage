import type { Metadata } from "next";
import Link from "next/link";
import { SolveStudio } from "../../solve/SolveStudio";
import { getSituation } from "../../../lib/situations";

const S = getSituation("procurement");

export const metadata: Metadata = {
  title: "Sealed bid for a contract — SAGE",
  description: S.decision,
};

export default function ProcurementSituation() {
  return (
    <div className="wrap situation-page">
      <p className="crumb">
        <Link href="/situations">All situations</Link>
      </p>
      <h1 className="surface-title">{S.name}</h1>
      <p className="surface-lede">{S.setting}</p>
      <SolveStudio fixedSituation="procurement" />
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
          One tender is one draw. If you bid against the same rival repeatedly,{" "}
          <Link href="/play">watch fifty rounds</Link> — the bidding rule that wins a single tender is often not the one
          that wins the year.
        </p>
      </section>
    </div>
  );
}
