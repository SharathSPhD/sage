"use client";

import Link from "next/link";
import { DynamicsTheater } from "../components/panels/DynamicsTheater";
import { PokePanel } from "../components/panels/PokePanel";
import { SimplexPortrait } from "../components/panels/SimplexPortrait";
import { SoftmaxCollapse } from "../components/panels/SoftmaxCollapse";

/* The guided on-ramp (Evolution-of-Trust pattern): one question, answered by
   playing the instruments in sequence. Prose recedes; the panels teach. */

function Beat({
  kicker,
  title,
  children,
  panel,
}: {
  kicker: string;
  title: string;
  children: React.ReactNode;
  panel?: React.ReactNode;
}) {
  return (
    <section style={{ margin: "4.5rem 0" }}>
      <div className="panel-label">{kicker}</div>
      <h2 style={{ fontSize: "1.5rem", margin: "0.3rem 0 0.8rem" }}>{title}</h2>
      <div style={{ color: "var(--text-dim)", maxWidth: "44rem", fontSize: "1.02rem" }}>{children}</div>
      {panel && <div style={{ marginTop: "1.4rem" }}>{panel}</div>}
    </section>
  );
}

export function StoryFlow() {
  return (
    <div className="wrap">
      <section className="hero" style={{ paddingBottom: "1rem" }}>
        <div className="kicker">the 5-minute tour</div>
        <h1>Is this market a landscape — or a whirlpool?</h1>
        <p className="lede">
          Some strategic systems are like water finding its level: everyone adjusts, things
          settle, and the settling point is a genuine equilibrium. Others never settle — strategy
          chases strategy in circles forever, like rock beats scissors beats paper beats rock.
          From the outside the two can look identical. This project builds meters that tell them
          apart. Five minutes, four instruments, real data at the end.
        </p>
      </section>

      <Beat
        kicker="beat 1 · the noise dial"
        title="First: nobody plays perfectly"
        panel={<SoftmaxCollapse />}
      >
        <p>
          Classical game theory says: pick the best action, with probability one. Real decision
          makers are payoff-<em>sensitive</em>, not payoff-perfect — better options get chosen
          more often, by an amount set by one dial, λ. Slide it: at the left everything is
          noise, at the far right you recover the classical answer. Everything else in this tour
          lives between those extremes.
        </p>
      </Beat>

      <Beat
        kicker="beat 2 · the loop"
        title="Everyone's noise responds to everyone's noise"
        panel={<SimplexPortrait />}
      >
        <p>
          Your mix of choices shapes their payoffs; their mix shapes yours. Click inside the
          triangle to drop a population anywhere and watch the loop run. On the coordination
          game it rolls downhill and <strong>stops</strong> — a landscape. On rock–paper–scissors
          it <strong>circulates</strong> — a whirlpool. Same math, opposite fates.
        </p>
      </Beat>

      <Beat
        kicker="beat 3 · the water"
        title="Still water or turning water — measured, not assumed"
        panel={<DynamicsTheater />}
      >
        <p>
          Here is the whole joint system as a lattice: every combination of choices, with
          probability as node size. In a landscape game the probability flow is still — every
          edge carries equal traffic both ways. In a whirlpool game the amber current runs one
          way around, forever, and the entropy-production meter reads it in nats. This is the
          live solver computing the exact stationary state — and the button estimates the same
          number from sampled trajectories alone, which is what makes real data reachable.
        </p>
      </Beat>

      <Beat
        kicker="beat 4 · the trick"
        title="You can measure this without knowing anyone's payoffs"
        panel={<PokePanel />}
      >
        <p>
          The payoffs are hidden in real systems. The meter that survives that: poke one
          player's incentives, read how the <em>other</em> moves; then poke the other, read the
          first. On a landscape the two cross-readings agree exactly — a deep symmetry from
          physics (Onsager reciprocity). On a whirlpool they disagree, and the disagreement is
          the measurement. Switch the game and watch the badge flip.
        </p>
      </Beat>

      <Beat kicker="beat 5 · the real world" title="So — what do real systems read?">
        <p>
          A real road network (Sioux Falls, with its actual demand table) reads{" "}
          <strong>exactly zero</strong> — traffic is a landscape, to machine precision, and you
          can toll any link to feel the symmetry:{" "}
          <Link href="/network">the network lab</Link>. A real power market (CAISO, July 2026)
          reads as a <strong>measurably driven cycle</strong> — the day-ahead price loop
          dissipates about 1.1 nats per day, concentrated exactly in the scarcity weeks, a
          finding that survived four null models and one retraction:{" "}
          <Link href="/markets">the market reading</Link>.
        </p>
        <p style={{ marginTop: "0.8rem" }}>
          Landscape or whirlpool is not a modelling <em>assumption</em>. It is a quantity you{" "}
          <em>measure</em> — and everything above regenerates from fixed seeds in the open
          repository, wins and retractions alike:{" "}
          <Link href="/findings">the anomaly log</Link>.
        </p>
        <div style={{ display: "flex", gap: "0.8rem", marginTop: "1.4rem", flexWrap: "wrap" }}>
          <Link href="/lab">
            <button data-primary="true">Open the full Lab</button>
          </Link>
          <Link href="/learn">
            <button>Read the ten explainers</button>
          </Link>
        </div>
      </Beat>
    </div>
  );
}
