import type { Metadata } from "next";
import Link from "next/link";
import { RoutingSolver } from "../../components/solvers/RoutingSolver";

export const metadata: Metadata = {
  title: "Traffic assignment — SAGE",
  description: "Toll one link on the Sioux Falls network and read link flows, total travel time and toll revenue.",
};

export default function RoutingPage() {
  return (
    <div className="wrap situation-page">
      <p className="crumb">
        <Link href="/situations">All problems</Link>
      </p>
      <h1 className="surface-title">Traffic assignment</h1>
      <p className="surface-lede">
        Every driver re-chooses a route after any change you make, so a toll moves traffic rather than removing it.
        Returns link flows, travel times, the network total, and what a toll did to all three.
      </p>
      <RoutingSolver />
      <section className="card next-step">
        <h2>In Python</h2>
        <p>
          <code>sq.RoutingProblem(network=&quot;sioux_falls&quot;, tolls=&#123;28: 5.0&#125;, precision=0.5).solve()</code> returns
          the same solution. An edge list of <code>(from, to, free_flow, capacity)</code> tuples works in place of the
          benchmark network.
        </p>
      </section>
      <p className="model-line">
        Sioux Falls: 24 nodes, 76 links, published capacities, free-flow times and demand table (TNTP). Top 12
        origin-destination pairs, three routes each.
      </p>
    </div>
  );
}
