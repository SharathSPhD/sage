import type { Metadata } from "next";
import Link from "next/link";
import { ProblemThumb } from "../components/ProblemThumb";
import { PROBLEMS } from "../../lib/catalogue";

export const metadata: Metadata = {
  title: "Problems",
  description:
    "Pricing, auctions, electricity offers, traffic assignment, budget allocation and arbitrary payoff tables — one solver, six pages.",
};

export default function ProblemsIndex() {
  return (
    <div className="wrap page">
      <h1 className="surface-title">Problem types</h1>
      <p className="surface-lede">
        Six shapes of problem over one solver. Each page takes your numbers and returns the quantity you came for, with
        a visual built for that problem rather than a generic chart.
      </p>
      <div className="gallery">
        {PROBLEMS.map((p) => (
          <Link key={p.id} href={p.href} className="card gallery-item">
            <ProblemThumb id={p.id} />
            <h2>{p.name}</h2>
            <p className="gallery-decision">{p.question}</p>
            <p className="gallery-setting">
              {p.visual} Returns {p.returns}.
            </p>
            <span className="badge">{p.endpoint}</span>
          </Link>
        ))}
      </div>
      <section className="card next-step">
        <h2>Something else</h2>
        <p>
          The Python library takes the same problems plus arbitrary payoff tensors —{" "}
          <a href="https://sharathsphd.github.io/sage/">docs</a>,{" "}
          <a href="https://github.com/SharathSPhD/sage">source</a>, and the{" "}
          <Link href="/api">API console</Link> for calling them over HTTP. To solve on numbers of your own, start at{" "}
          <Link href="/data">bring your own data</Link>.
        </p>
      </section>
    </div>
  );
}
