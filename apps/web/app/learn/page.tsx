import Link from "next/link";
import { listExplainers } from "../../lib/theory";

export default function LearnIndex() {
  const explainers = listExplainers();
  const planned = [
    "02 · The fixed point",
    "03 · QRE vs mixed Nash",
    "04 · MaxEnt",
    "05 · Gibbs and potential games",
    "06 · Detailed balance and currents",
    "08 · Elasticity vs λ",
    "10 · The same machinery everywhere",
  ];
  return (
    <div>
      <h1>Learn</h1>
      <p>
        Ten explainers, authored once in <code>docs/theory/</code> and rendered here and on the
        docs site. Written so far:
      </p>
      <ul style={{ lineHeight: 2 }}>
        {explainers.map((e) => (
          <li key={e.slug}>
            <Link href={`/learn/${e.slug}`}>{e.title}</Link>
          </li>
        ))}
      </ul>
      <p style={{ color: "#777" }}>In the pipeline: {planned.join(" · ")}</p>
    </div>
  );
}
