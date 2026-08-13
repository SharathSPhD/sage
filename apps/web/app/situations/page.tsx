import type { Metadata } from "next";
import Link from "next/link";
import { ENGINE_SITUATIONS, SITUATIONS } from "../../lib/situations";

export const metadata: Metadata = {
  title: "Situations — SAGE",
  description:
    "Five situations you can actually play: weekly pricing, a sealed bid, a road charge, splitting a budget, and picking a standard.",
};

const ALL = [
  ...SITUATIONS.map((s) => ({
    id: s.id,
    name: s.name,
    decision: s.decision,
    setting: s.setting,
    sourceNote: s.sourceNote,
    illustrative: s.illustrative,
    href: s.href,
  })),
  ...ENGINE_SITUATIONS,
];

export default function SituationsIndex() {
  return (
    <div className="wrap" style={{ paddingTop: "2.4rem" }}>
      <h1 className="surface-title">Situations</h1>
      <p className="surface-lede">
        Each one is a real decision with numbers you can change. Pick the one closest to yours; the machinery
        underneath is the same.
      </p>
      <div className="gallery">
        {ALL.map((s) => (
          <Link key={s.id} href={s.href} className="card gallery-item">
            <h2>{s.name}</h2>
            <p className="gallery-decision">{s.decision}</p>
            <p className="gallery-setting">{s.setting}</p>
            <span className="badge" data-tone={s.illustrative ? undefined : "ok"}>
              {s.illustrative ? "illustrative numbers" : "measured numbers"}
            </span>
          </Link>
        ))}
      </div>
      <section className="card next-step">
        <h2>Nothing here matches</h2>
        <p>
          The situations above are two-sided, repeated, and small enough to lay out on a table. If yours is that shape
          but the numbers differ, change them on any page — they are all live. If it is a different shape,{" "}
          <a href="https://github.com/SharathSPhD/sage">the library</a> takes an arbitrary payoff table, and{" "}
          <Link href="/diagnose">the fit check</Link> will tell you whether your data supports this kind of answer at
          all.
        </p>
      </section>
    </div>
  );
}
