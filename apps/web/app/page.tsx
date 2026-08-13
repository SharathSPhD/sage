import Link from "next/link";
import { LiveDemo } from "./components/LiveDemo";
import { ProblemThumb } from "./components/ProblemThumb";
import { PROBLEMS } from "../lib/catalogue";

export default function Home() {
  return (
    <div className="wrap">
      <section className="landing-hero">
        <div>
          <p className="eyebrow">strataq · quantal response equilibria</p>
          <h1>Decide well when the other side is deciding too.</h1>
          <p className="lede">
            Prices, tenders, electricity offers, road tolls and budget splits are all the same shape of problem: your
            best move depends on theirs, and theirs on yours. Put in your costs, your grid and how demand behaves, and
            get the move to make, what it earns, and what would change it.
          </p>
          <div className="hero-actions">
            <Link href="/solve" className="btn" data-primary="true">
              Solve a problem
            </Link>
            <Link href="/demos" className="btn">
              See the demos
            </Link>
            <Link href="/data" className="btn">
              Bring your own data
            </Link>
            <Link href="/api" className="btn">
              Use the API
            </Link>
          </div>
          <div className="hero-stats">
            <div className="hero-stat">
              <div className="panel-label">Problem types</div>
              <div className="reading">6</div>
            </div>
            <div className="hero-stat">
              <div className="panel-label">Typical solve</div>
              <div className="reading">
                &lt;50<span className="unit">ms</span>
              </div>
            </div>
            <div className="hero-stat">
              <div className="panel-label">Every answer via</div>
              <div className="reading">
                <Link href="/api">one API</Link>
              </div>
            </div>
          </div>
        </div>
        <LiveDemo />
      </section>

      <section className="card home-demos" aria-labelledby="demos-heading">
        <div>
          <h2 id="demos-heading">Four demos that do not exist anywhere else</h2>
          <p>
            Have your own rationality fitted by maximum likelihood while you play rock&ndash;paper&ndash;scissors; watch
            probability circulate on the nine joint states of a game and the entropy-production counter read exactly zero
            when it should; guess what real experimental subjects did; and put a game on the plane next to Sioux Falls,
            Dominick&apos;s and CAISO. Every one runs its mathematics live in the page.
          </p>
        </div>
        <Link href="/demos" className="btn" data-primary="true">
          Open the demos
        </Link>
      </section>

      <section className="home-gallery" aria-labelledby="gallery-heading">
        <div className="section-head">
          <h2 id="gallery-heading">Six problems, one solver</h2>
          <p>
            Each has its own page, its own inputs and a visual built for it. <Link href="/situations">See them all</Link>
          </p>
        </div>
        <div className="gallery">
          {PROBLEMS.map((p) => (
            <Link key={p.id} href={p.href} className="card gallery-item">
              <ProblemThumb id={p.id} />
              <h3>{p.name}</h3>
              <p className="gallery-decision">{p.question}</p>
              <p className="gallery-setting">{p.visual}</p>
              <span className="badge">{p.endpoint}</span>
            </Link>
          ))}
        </div>
      </section>

      <section className="home-gallery" aria-labelledby="who-heading">
        <div className="section-head">
          <h2 id="who-heading">Who this is for</h2>
        </div>
        <div className="entry-grid">
          <div className="card entry-card">
            <h3>Pricing and revenue teams</h3>
            <p>
              You have a cost, a shelf and a competitor. You want the price, the margin it leaves, and the honest answer
              to what happens if their cost drops ten cents.
            </p>
          </div>
          <div className="card entry-card">
            <h3>Bid and tender desks</h3>
            <p>
              You are one of a handful of credible bidders. You want the bid, the chance it wins, and what the surplus
              looks like a step either side of it.
            </p>
          </div>
          <div className="card entry-card">
            <h3>Market and network operators</h3>
            <p>
              Generator offers into a uniform-price market, or traffic over a network with a toll on one link. You want
              the system total, not just your own line.
            </p>
          </div>
          <div className="card entry-card">
            <h3>Analysts and researchers</h3>
            <p>
              Everything the app does is one HTTP call, and the same call from Python returns the same numbers. Fit the
              precision to your own observed choices first.
            </p>
          </div>
        </div>
      </section>

      <section className="home-gallery" aria-labelledby="entry-heading">
        <div className="section-head">
          <h2 id="entry-heading">Start here</h2>
        </div>
        <div className="entry-grid">
          <Link href="/solve" className="card entry-card">
            <span className="entry-icon" aria-hidden>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9">
                <path d="M4 17h16M4 12h10M4 7h6" strokeLinecap="round" />
              </svg>
            </span>
            <h3>Solve</h3>
            <p>The studio: pick a problem type, put in your numbers, watch the answer move as you drag.</p>
            <span className="entry-go">Open the studio →</span>
          </Link>
          <Link href="/data" className="card entry-card">
            <span className="entry-icon" aria-hidden>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9">
                <path d="M12 16V4m0 0L8 8m4-4 4 4M5 20h14" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </span>
            <h3>Bring your data</h3>
            <p>Upload or paste a CSV, map the columns, estimate what the data can pin down, then solve on it.</p>
            <span className="entry-go">Load a file →</span>
          </Link>
          <Link href="/api" className="card entry-card">
            <span className="entry-icon" aria-hidden>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9">
                <path d="M9 7 4 12l5 5M15 7l5 5-5 5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </span>
            <h3>API</h3>
            <p>Every endpoint, its schema, a live console and copyable curl and Python — read from the live service.</p>
            <span className="entry-go">Open the console →</span>
          </Link>
          <Link href="/learn" className="card entry-card">
            <span className="entry-icon" aria-hidden>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9">
                <path d="M4 5.5A1.5 1.5 0 0 1 5.5 4H11v16H5.5A1.5 1.5 0 0 1 4 18.5zM20 5.5A1.5 1.5 0 0 0 18.5 4H13v16h5.5a1.5 1.5 0 0 0 1.5-1.5z" strokeLinejoin="round" />
              </svg>
            </span>
            <h3>Learn</h3>
            <p>What a precision is, why a distribution beats a point forecast, and where the method stops working.</p>
            <span className="entry-go">Read the explainers →</span>
          </Link>
        </div>
      </section>

      <section className="card home-play">
        <h2>Test a rule over repeated rounds</h2>
        <p>
          A single solve answers one round. The backtest runs a hundred against a simulated rival and compares the
          solver&apos;s move with cost-plus, matching, best-reply-to-last and always-Nash on the same draws.
        </p>
        <Link href="/play" className="btn">
          Open the backtest
        </Link>
      </section>
    </div>
  );
}
