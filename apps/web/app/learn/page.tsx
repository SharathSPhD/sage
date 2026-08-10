import Link from "next/link";
import { listExplainers } from "../../lib/theory";
import { firstParagraph } from "../../lib/markdown";

export const metadata = { title: "Learn — SAGE Labs" };

export default function LearnIndex() {
  const explainers = listExplainers();
  return (
    <div className="wrap-narrow" style={{ paddingTop: "2.2rem" }}>
      <h1 style={{ marginBottom: "0.3rem" }}>Learn</h1>
      <p style={{ color: "var(--text-dim)", marginTop: 0 }}>
        Ten short explainers, in reading order. They are the single source for both this app and
        the docs site, and they include the strongest objection to the whole approach — number
        nine — because an instrument you cannot argue with is not an instrument.
      </p>
      <div className="learn-index">
        {explainers.map((e, i) => (
          <Link key={e.slug} href={`/learn/${e.slug}`} className="card learn-item">
            <span className="num">{String(i + 1).padStart(2, "0")}</span>
            <span>
              <h3>{e.title}</h3>
              <p>{firstParagraph(e.markdown)}…</p>
            </span>
          </Link>
        ))}
      </div>
    </div>
  );
}
