import Link from "next/link";
import { LiveStrip } from "./components/LiveStrip";

const FEATURES = [
  {
    href: "/lab",
    title: "Lab",
    body: "Pick a game, slide λ, watch every meter respond live: equilibrium mix, reciprocity defect, dissipation, distance to criticality — computed by the deployed solver, not the browser.",
  },
  {
    href: "/phase",
    title: "Phase map",
    body: "The α×λ plane from 9,900 solved games: where dissipation lives, where reciprocity breaks, and the supercritical wedge nobody expected.",
  },
  {
    href: "/learn",
    title: "Learn",
    body: "Ten short explainers, honestly written — including the strongest objection to the whole approach and what survives it.",
  },
  {
    href: "https://sharathsphd.github.io/sage/progress/",
    title: "Gate dashboard",
    body: "Every claim's confidence tier, every unit's gate state, every red-team objection and its disposition. The process is the product.",
  },
] as const;

export default function Home() {
  return (
    <div className="wrap">
      <section className="hero">
        <div className="kicker">ThermoQRE · strategic thermodynamics</div>
        <h1>Instruments for strategic systems</h1>
        <p className="lede">
          SAGE Labs is a research instrument, not a demo: a susceptibility meter, a reciprocity
          meter, an entropy-production meter and a phase locator for quantal-response equilibria.
          Each is calibrated on games where the correct reading is known — a real road network
          reads exactly zero, rock–paper–scissors reads loudly — then pointed at systems where
          nobody knows the answer.
        </p>
        <div style={{ display: "flex", gap: "0.8rem", flexWrap: "wrap" }}>
          <Link href="/lab">
            <button data-primary="true">Open the Lab</button>
          </Link>
          <Link href="/learn">
            <button>Start with the ideas</button>
          </Link>
        </div>
      </section>

      <LiveStrip />

      <section className="feature-grid">
        {FEATURES.map((f) =>
          f.href.startsWith("/") ? (
            <Link key={f.title} href={f.href} className="card">
              <h3>{f.title} →</h3>
              <p>{f.body}</p>
            </Link>
          ) : (
            <a key={f.title} href={f.href} className="card">
              <h3>{f.title} ↗</h3>
              <p>{f.body}</p>
            </a>
          ),
        )}
      </section>

      <section style={{ margin: "3rem 0" }}>
        <div className="panel-label">The calibration bracket</div>
        <div className="calibration-strip">
          <div className="card">
            <div className="panel-label">Sioux Falls road network · α = 0</div>
            <div className="reading">
              ℛ = 5.7×10⁻¹⁷<span className="unit">≈ 0</span>
            </div>
            <p style={{ color: "var(--text-dim)", fontSize: "0.85rem", margin: "0.5rem 0 0" }}>
              Real traffic-assignment data. A potential system: strategic feedback preserves
              reciprocity, the meter must read zero — and does, to machine precision.
            </p>
          </div>
          <div className="card">
            <div className="panel-label">Rock–paper–scissors · α = 1</div>
            <div className="reading" data-tone="warn">
              ℛ = 0.87
            </div>
            <p style={{ color: "var(--text-dim)", fontSize: "0.85rem", margin: "0.5rem 0 0" }}>
              Pure strategic cycling. Broken reciprocity, positive entropy production, circulating
              probability current — the instruments read loudly, as they must.
            </p>
          </div>
          <div className="card">
            <div className="panel-label">Colonel Blotto · α = 0.69</div>
            <div className="reading" data-tone="neutral">
              ℛ = 0.12
            </div>
            <p style={{ color: "var(--text-dim)", fontSize: "0.85rem", margin: "0.5rem 0 0" }}>
              The middle of the dial: a mixed game reads between the anchors, ordered by its
              harmonic fraction. Spearman ρ(ℛ, α) = 0.98 over 2,000 games.
            </p>
          </div>
        </div>
      </section>

      <section className="card" style={{ margin: "3rem 0", borderColor: "var(--border-bright)" }}>
        <div className="panel-label">What this is measuring</div>
        <p style={{ color: "var(--text-dim)", margin: 0 }}>
          Logit quantal response puts strategic interaction in exact correspondence with
          statistical mechanics: the rationality parameter λ is an inverse temperature, the
          log-partition function is a cumulant generating function, and a game&apos;s decomposition
          into <em>potential</em> and <em>harmonic</em> parts (α is the harmonic fraction)
          determines whether its dynamics relax to equilibrium or circulate forever, dissipating.
          The reciprocity defect ℛ turns that hidden structure into something estimable from
          cross-price pass-through asymmetry alone — no payoffs needed. Three genuine discoveries
          so far, each with its falsifier on record:{" "}
          <a href="https://sharathsphd.github.io/sage/">F-0004, F-0006, F-0007</a>.
        </p>
      </section>
    </div>
  );
}
