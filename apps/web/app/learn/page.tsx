import Link from "next/link";
import { listExplainers } from "../../lib/theory";

export default function LearnIndex() {
  const explainers = listExplainers();
  return (
    <div>
      <h1>Learn</h1>
      <p>
        Ten explainers, authored once in <code>docs/theory/</code> and rendered here and on the
        docs site.
      </p>
      <ul style={{ lineHeight: 2 }}>
        {explainers.map((e) => (
          <li key={e.slug}>
            <Link href={`/learn/${e.slug}`}>{e.title}</Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
