import Link from "next/link";
import { Workbench } from "./components/solvers/Workbench";
import { PROBLEMS } from "../lib/catalogue";

export default function Home() {
  return (
    <div className="wrap">
      <section className="hero">
        <h1>Solve the problem where the other side is optimising too.</h1>
        <p className="lede">
          Enter costs, a grid and a demand model; get the price, bid, offer or split to set, what it earns, and how it
          moves when an input does. Answers come from the same solver the Python library calls.
        </p>
      </section>

      <Workbench />

      <section className="home-gallery" aria-labelledby="gallery-heading">
        <h2 id="gallery-heading">Problem types</h2>
        <div className="gallery">
          {PROBLEMS.map((p) => (
            <Link key={p.id} href={p.href} className="card gallery-item">
              <h3>{p.name}</h3>
              <p className="gallery-decision">{p.question}</p>
              <p className="gallery-setting">Returns {p.returns}</p>
              <span className="badge">{p.endpoint}</span>
            </Link>
          ))}
        </div>
      </section>

      <section className="card home-play">
        <h2>Test a rule over repeated rounds</h2>
        <p>
          A single solve answers one round. The backtest runs a hundred against a simulated rival and compares the
          solver&apos;s move with cost-plus, matching, best-reply-to-last and always-Nash on the same draws.
        </p>
        <Link href="/play">
          <button data-primary="true">Open the backtest</button>
        </Link>
      </section>
    </div>
  );
}
