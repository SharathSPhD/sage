import Link from "next/link";
import { listExplainers } from "../../lib/theory";
import { firstParagraph } from "../../lib/markdown";
import { TAKEAWAYS } from "./takeaways";

export const metadata = {
  title: "Learn",
  description:
    "Ten short explainers, each ending in what you would do differently on Monday. Including the strongest objection to the whole approach.",
};

export default function LearnIndex() {
  const explainers = listExplainers();
  return (
    <div className="wrap-narrow" style={{ paddingTop: "2.4rem" }}>
      <h1 className="surface-title">Learn</h1>
      <p className="surface-lede">
        Ten short pieces, in order. Each one ends with what it changes about a decision — if it does not change one,
        it is not worth your time. Number nine is the strongest argument against this whole approach.
      </p>
      <div className="learn-index">
        {explainers.map((e, i) => (
          <Link key={e.slug} href={`/learn/${e.slug}`} className="card learn-item">
            <span className="num">{String(i + 1).padStart(2, "0")}</span>
            <span>
              <h3>{e.title}</h3>
              <p>{TAKEAWAYS[e.slug]?.soWhat ?? `${firstParagraph(e.markdown)}…`}</p>
            </span>
          </Link>
        ))}
      </div>
    </div>
  );
}
