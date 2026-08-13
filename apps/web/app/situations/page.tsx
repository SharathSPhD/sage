import type { Metadata } from "next";
import Link from "next/link";
import { PROBLEMS } from "../../lib/catalogue";

export const metadata: Metadata = {
  title: "Problems — SAGE",
  description:
    "Pricing, auctions, electricity offers, traffic assignment, budget allocation and arbitrary payoff tables.",
};

export default function ProblemsIndex() {
  return (
    <div className="wrap" style={{ paddingTop: "2.4rem" }}>
      <h1 className="surface-title">Problems</h1>
      <p className="surface-lede">
        Six problem types, one solver. Each page takes your numbers and returns the quantity you came for.
      </p>
      <div className="gallery">
        {PROBLEMS.map((p) => (
          <Link key={p.id} href={p.href} className="card gallery-item">
            <h2>{p.name}</h2>
            <p className="gallery-decision">{p.question}</p>
            <p className="gallery-setting">Returns {p.returns}</p>
            <span className="badge">{p.endpoint}</span>
          </Link>
        ))}
      </div>
      <section className="card next-step">
        <h2>Something else</h2>
        <p>
          The Python library takes the same problems plus arbitrary payoff tensors —{" "}
          <a href="https://sharathsphd.github.io/sage/">docs</a>,{" "}
          <a href="https://github.com/SharathSPhD/sage">source</a>. To check whether this class of model fits your own
          series before you use it, the <Link href="/diagnose">fit check</Link> takes a CSV.
        </p>
      </section>
    </div>
  );
}
