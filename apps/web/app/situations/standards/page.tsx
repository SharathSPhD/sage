import type { Metadata } from "next";
import Link from "next/link";
import { SolveStudio } from "../../solve/SolveStudio";
import { getSituation } from "../../../lib/situations";

const S = getSituation("standards");

export const metadata: Metadata = {
  title: "Which standard to back — SAGE",
  description: S.decision,
};

export default function StandardsSituation() {
  return (
    <div className="wrap situation-page">
      <p className="crumb">
        <Link href="/situations">All situations</Link>
      </p>
      <h1 className="surface-title">{S.name}</h1>
      <p className="surface-lede">{S.setting}</p>
      <SolveStudio fixedSituation="standards" />
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
    </div>
  );
}
