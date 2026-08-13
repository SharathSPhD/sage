import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "How it works, and how we know — SAGE",
  description:
    "The evidence under the solutions: the fit check, the phase map, the anomaly log, market readings, and the raw solver bench.",
};

const PAGES = [
  {
    href: "/diagnose",
    title: "Does this model fit my market?",
    body: "Drop in a price or flow series and find out whether the assumptions behind every answer on this site hold for your data — and, when they do not, exactly which quantity your data fails to pin down.",
  },
  {
    href: "/lab",
    title: "The raw solver bench",
    body: "Any payoff table, any rationality setting, every quantity the library computes. No practitioner framing: this is the engine with the covers off.",
  },
  {
    href: "/phase",
    title: "The map of every game",
    body: "9,900 solved games arranged by structure and rationality, showing which regions circulate forever and which settle. Where a situation sits on this map is what determines whether timing or structure is the thing to fix.",
  },
  {
    href: "/findings",
    title: "The anomaly log",
    body: "Every unexpected result, including the retractions and the hypotheses that died. Each entry carries the seed that regenerates it.",
  },
  {
    href: "/markets",
    title: "Electricity market reading",
    body: "CAISO day-ahead and real-time prices through the same machinery, and what the difference between the two says about who is driving whom.",
  },
  {
    href: "/story",
    title: "The five-minute tour",
    body: "How this started, what was expected, and what actually turned up. Written before the app was about solutions; kept because the record should not be edited after the fact.",
  },
  {
    href: "/tools",
    title: "Two single-purpose panels",
    body: "The reciprocity and irreversibility panels on their own, for pasting a series into. Superseded by the fit check, kept working.",
  },
];

export default function ResearchIndex() {
  return (
    <div className="wrap" style={{ paddingTop: "2.4rem" }}>
      <h1 className="surface-title">How it works, and how we know</h1>
      <p className="surface-lede">
        The pages that produce answers are elsewhere. These are the ones that say why the answers can be trusted, where
        the method breaks, and what it got wrong on the way.
      </p>
      <div className="gallery">
        {PAGES.map((p) => (
          <Link key={p.href} href={p.href} className="card gallery-item">
            <h2>{p.title}</h2>
            <p className="gallery-setting">{p.body}</p>
          </Link>
        ))}
      </div>
      <section className="card next-step">
        <h2>The full record</h2>
        <p>
          Everything regenerates from fixed seeds in{" "}
          <a href="https://github.com/SharathSPhD/sage">the open repository</a>. The{" "}
          <a href="https://sharathsphd.github.io/sage/progress/">gate dashboard</a> lists every claim, its confidence
          tier, and every adversarial review that closed it. <Link href="/learn">The explainers</Link> cover the theory
          from the ground up, including the strongest objection to the whole approach.
        </p>
      </section>
    </div>
  );
}
