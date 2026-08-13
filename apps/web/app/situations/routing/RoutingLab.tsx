"use client";

/* Where to put the toll. Real network, real demand table, real solver:
 * the TNTP Sioux Falls benchmark, drawn from its published node coordinates,
 * with link flows from the deployed route-choice solver on every change.
 * Previously /network.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Term } from "../../components/Term";

// TNTP SiouxFalls_node.tntp coordinates (lon, lat) — drawing layout only;
// every flow on screen comes back from the solver.
const NODES: [number, number][] = [
  [-96.7704, 43.6128], [-96.7113, 43.6058], [-96.7743, 43.573], [-96.7472, 43.5637],
  [-96.7316, 43.564], [-96.7116, 43.5876], [-96.6934, 43.5638], [-96.7114, 43.5623],
  [-96.7312, 43.5486], [-96.7314, 43.5453], [-96.7468, 43.5441], [-96.7801, 43.5439],
  [-96.7934, 43.4907], [-96.751, 43.5293], [-96.7315, 43.5294], [-96.7114, 43.5467],
  [-96.7114, 43.5413], [-96.6941, 43.5467], [-96.7113, 43.5296], [-96.7112, 43.5153],
  [-96.731, 43.5105], [-96.7312, 43.5149], [-96.7509, 43.5149], [-96.7492, 43.5032],
];

interface Link {
  from: number;
  to: number;
  free_flow: number;
  capacity: number;
}

interface SUE {
  link_flows: number[];
  link_costs: number[];
  total_travel_time: number;
  beckmann: number;
  residual: number;
}

const W = 560;
const H = 640;

function project(): (lon: number, lat: number) => [number, number] {
  const lons = NODES.map((n) => n[0]);
  const lats = NODES.map((n) => n[1]);
  const [lo0, lo1] = [Math.min(...lons), Math.max(...lons)];
  const [la0, la1] = [Math.min(...lats), Math.max(...lats)];
  const pad = 45;
  return (lon, lat) => [
    pad + ((lon - lo0) / (lo1 - lo0)) * (W - 2 * pad),
    pad + ((la1 - lat) / (la1 - la0)) * (H - 2 * pad),
  ];
}

export function RoutingLab() {
  const [links, setLinks] = useState<Link[]>([]);
  const [spread, setSpread] = useState(0.5);
  const [tollLink, setTollLink] = useState<number | null>(null);
  const [toll, setToll] = useState(10);
  const [sue, setSue] = useState<SUE | null>(null);
  const [baseline, setBaseline] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const toXY = useMemo(project, []);

  useEffect(() => {
    fetch("/api/v1/domains/sioux_falls/network")
      .then((r) => r.json())
      .then((d) => setLinks(d.links ?? []))
      .catch(() => undefined);
  }, []);

  const solve = useCallback(() => {
    abortRef.current?.abort();
    const ctl = new AbortController();
    abortRef.current = ctl;
    setBusy(true);
    const tolls = tollLink === null ? undefined : { [tollLink]: toll };
    fetch("/api/v1/domains/sioux_falls/sue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ theta: spread, tolls }),
      signal: ctl.signal,
    })
      .then((r) => r.json())
      .then((d: SUE) => {
        if (ctl.signal.aborted) return;
        setSue(d);
        if (tollLink === null) setBaseline(d.total_travel_time);
        setBusy(false);
      })
      .catch(() => !ctl.signal.aborted && setBusy(false));
  }, [spread, tollLink, toll]);

  useEffect(() => {
    const t = setTimeout(solve, 350);
    return () => clearTimeout(t);
  }, [solve]);

  const maxFlow = sue ? Math.max(...sue.link_flows, 1) : 1;
  const delta = sue && baseline !== null && tollLink !== null ? sue.total_travel_time - baseline : null;
  const worst =
    sue && links.length
      ? sue.link_flows
          .map((f, i) => ({ i, vc: f / Math.max(links[i]?.capacity ?? 1, 1) }))
          .sort((a, b) => b.vc - a.vc)[0]
      : null;

  return (
    <div className="studio">
      <section className="answer" aria-labelledby="route-answer">
        <div className="answer-head">
          <h2 id="route-answer">Do this</h2>
          <span className="badge" data-tone="ok">
            solved by the deployed solver on real network data
          </span>
        </div>
        {tollLink === null ? (
          <>
            <p className="answer-verb">
              {worst && links[worst.i]
                ? `Start with the junction ${links[worst.i].from} → ${links[worst.i].to} link — it is running at ${(100 * worst.vc).toFixed(0)}% of capacity.`
                : "Pick a link on the map to price it."}
            </p>
            <p className="lead-note">
              Click any road to charge for it. The whole city re-routes; the number that matters is the total, not the
              link you touched.
            </p>
          </>
        ) : (
          <>
            <p className="answer-verb">
              {delta === null
                ? "Solving…"
                : delta < 0
                  ? `Charge ${toll} minutes-equivalent on ${links[tollLink]?.from} → ${links[tollLink]?.to}: the city gets ${Math.abs(delta).toFixed(0)} vehicle-minutes back.`
                  : `Do not charge on ${links[tollLink]?.from} → ${links[tollLink]?.to}: at ${toll} minutes-equivalent it costs the city ${delta.toFixed(0)} extra vehicle-minutes.`}
            </p>
            <p className="lead-note">
              Traffic does not disappear when you price a road; it moves. Whether that helps depends entirely on where
              it moves to.
            </p>
          </>
        )}
      </section>

      <div className="route-cols">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="route-map"
          style={{ opacity: busy ? 0.65 : 1 }}
          role="img"
          aria-label="Sioux Falls road network. Line width is traffic volume; colour is how close each road is to capacity. Use the list below the map to charge for a road with the keyboard."
        >
          {links.map((l, i) => {
            const [x1, y1] = toXY(...NODES[l.from - 1]);
            const [x2, y2] = toXY(...NODES[l.to - 1]);
            const flow = sue ? sue.link_flows[i] : 0;
            const vc = flow / Math.max(l.capacity, 1);
            const hue = vc > 0.9 ? "var(--red)" : vc > 0.55 ? "var(--amber)" : "var(--accent)";
            const isTolled = tollLink === i;
            return (
              <line
                key={i}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke={isTolled ? "var(--text)" : hue}
                strokeWidth={0.6 + 5 * (flow / maxFlow)}
                opacity={isTolled ? 1 : 0.75}
                strokeDasharray={isTolled ? "5 3" : undefined}
                style={{ cursor: "pointer" }}
                onClick={() => setTollLink(isTolled ? null : i)}
              />
            );
          })}
          {NODES.map(([lon, lat], k) => {
            const [x, y] = toXY(lon, lat);
            return (
              <g key={k}>
                <circle cx={x} cy={y} r="7.5" fill="var(--panel)" stroke="var(--border-bright)" />
                <text x={x} y={y + 3} textAnchor="middle" fontSize="7.5" fill="var(--text-dim)" fontFamily="var(--mono)">
                  {k + 1}
                </text>
              </g>
            );
          })}
        </svg>

        <div>
          <section className="card controls">
            <h3>Your numbers</h3>
            <div className="knob-grid">
              <label className="knob">
                <span className="knob-label">
                  Which road to charge for
                  <output aria-hidden="true">
                    {tollLink === null ? "none" : `${links[tollLink]?.from} → ${links[tollLink]?.to}`}
                  </output>
                </span>
                <select
                  aria-label="Which road to charge for"
                  value={tollLink === null ? "" : String(tollLink)}
                  onChange={(e) => setTollLink(e.target.value === "" ? null : Number(e.target.value))}
                >
                  <option value="">No charge anywhere</option>
                  {links.map((l, i) => (
                    <option key={i} value={i}>
                      {l.from} → {l.to}
                    </option>
                  ))}
                </select>
              </label>
              <label className="knob">
                <span className="knob-label">
                  Charge, in minutes drivers would pay to avoid it
                  <output aria-hidden="true">{toll.toFixed(0)}</output>
                </span>
                <input
                  type="range"
                  aria-label="Charge, in minutes drivers would pay to avoid it"
                  min={0}
                  max={50}
                  step={1}
                  value={toll}
                  onChange={(e) => setToll(Number(e.target.value))}
                  disabled={tollLink === null}
                />
              </label>
              <label className="knob">
                <span className="knob-label">
                  How well drivers know the fastest route
                  <output aria-hidden="true">{spread.toFixed(2)}</output>
                </span>
                <input
                  type="range"
                  aria-label="How well drivers know the fastest route"
                  min={0.05}
                  max={3}
                  step={0.05}
                  value={spread}
                  onChange={(e) => setSpread(Number(e.target.value))}
                />
                <span className="knob-help">
                  Low spreads traffic over every plausible route; high piles it all onto the fastest.{" "}
                  <Term
                    term="Written θ"
                    explain="θ, the route-choice precision in a stochastic user equilibrium — the population version of λ."
                  />
                </span>
              </label>
            </div>
          </section>

          {sue && (
            <section className="card">
              <h3>What it costs the city</h3>
              <table className="alt-table">
                <tbody>
                  <tr>
                    <th scope="row">Total time on the network</th>
                    <td>{sue.total_travel_time.toFixed(0)} vehicle-minutes</td>
                  </tr>
                  {delta !== null && (
                    <tr>
                      <th scope="row">Against no charge at all</th>
                      <td className={delta > 0 ? "alt-cost" : "alt-gain"}>
                        {delta > 0 ? "+" : "−"}
                        {Math.abs(delta).toFixed(0)}
                      </td>
                    </tr>
                  )}
                  <tr>
                    <th scope="row">Busiest road, share of capacity</th>
                    <td>{worst ? `${(100 * worst.vc).toFixed(0)}%` : "—"}</td>
                  </tr>
                </tbody>
              </table>
            </section>
          )}
        </div>
      </div>

      <details className="under-the-hood">
        <summary>The numbers behind this</summary>
        <ul>
          <li>
            Sioux Falls, the standard 24-junction / 76-link benchmark, with its published demand table. Top 12
            origin-destination pairs, three routes each.
          </li>
          <li>
            Flows are a stochastic user equilibrium from the deployed Fisk–Newton solver; the reported residual is the
            solver&apos;s own convergence check, {sue ? sue.residual.toExponential(1) : "—"}.
          </li>
          <li>
            <Term
              term="Reciprocity defect ℛ"
              explain="How far the two sides' cross-responses fail to mirror each other. Route choice over congestion costs is an exact potential game, so this is zero up to floating point — 5.7×10⁻¹⁷ at the project's gate."
            />{" "}
            is zero here, which is what licenses the mirror rule above.
          </li>
        </ul>
      </details>
    </div>
  );
}
