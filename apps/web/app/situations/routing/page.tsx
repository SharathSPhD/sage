import type { Metadata } from "next";
import Link from "next/link";
import { RoutingLab } from "./RoutingLab";

export const metadata: Metadata = {
  title: "Where to put the toll — SAGE",
  description: "Charge for one road on a real city network and see whether the city actually gets time back.",
};

export default function RoutingSituation() {
  return (
    <div className="wrap situation-page">
      <p className="crumb">
        <Link href="/situations">All situations</Link>
      </p>
      <h1 className="surface-title">Where to put the toll</h1>
      <p className="surface-lede">
        You can price or close exactly one road. Every driver then picks their own route again, and the queue you
        removed reappears somewhere else. The question is whether the city is better off after they have all moved.
      </p>
      <RoutingLab />
      <section className="try-this" aria-labelledby="try-heading">
        <h2 id="try-heading">Three things worth trying</h2>
        <div className="try-grid">
          <article className="card">
            <h3>Charging for the busiest road can make things worse</h3>
            <p>
              Pick the road running closest to capacity and charge 30. Total time on the network often goes up: the
              traffic you pushed off it lands on roads with less headroom. Congestion is not a property of one link.
            </p>
          </article>
          <article className="card">
            <h3>You only ever need to test half your options</h3>
            <p>
              A charge on road A moves traffic on road B by exactly as much as the same charge on B would move A. That
              symmetry is a property of route choice, not a coincidence, and it halves the number of trials any
              scheme needs.
            </p>
          </article>
          <article className="card">
            <h3>Better navigation is not automatically better traffic</h3>
            <p>
              Slide driver knowledge from 0.3 to 3. Everyone finds the fastest route, and total time on the network
              gets worse, not better — because the fastest route for each driver is not the fastest set of routes for
              all of them.
            </p>
          </article>
        </div>
      </section>
      <p className="source-note" data-illustrative={false}>
        <strong>Measured numbers.</strong> The Sioux Falls benchmark network with its published link capacities, free-flow
        times and demand table. Every flow on this page comes back from the deployed solver; nothing is precomputed.
      </p>
    </div>
  );
}
