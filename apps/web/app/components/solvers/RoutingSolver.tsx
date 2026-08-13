"use client";

/* Traffic assignment on the Sioux Falls benchmark, with a toll on one link.
 *
 * POST /v1/solve/routing — body mirrors sq.RoutingProblem(network=, tolls=,
 * precision=, max_od=). Link flows, travel times, total cost and the toll
 * effect all come back from that call; the map only supplies coordinates.
 */

import { useEffect, useMemo, useState } from "react";
import { useSolve, useSweep, type RoutingSolution } from "../../../lib/problems";
import { Answer, Controls, Field, Figure, ModelLine, Sweep, count, money0, num } from "./ui";

// TNTP SiouxFalls_node.tntp coordinates. Drawing only — every flow on screen
// comes back from the solver.
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

const TOLL_POINTS = [0, 5, 10, 20, 35, 50];

export function RoutingSolver() {
  const [links, setLinks] = useState<Link[]>([]);
  const [precision, setPrecision] = useState(0.5);
  const [tollLink, setTollLink] = useState<number | null>(null);
  const [toll, setToll] = useState(10);
  const toXY = useMemo(project, []);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/v1/domains/sioux_falls/network", { signal: controller.signal })
      .then((r) => r.json())
      .then((d) => setLinks(d.links ?? []))
      .catch(() => undefined);
    return () => controller.abort();
  }, []);

  const body = useMemo(
    () => ({
      network: "sioux_falls",
      precision,
      max_od: 12,
      ...(tollLink === null ? {} : { tolls: { [tollLink]: toll } }),
    }),
    [precision, tollLink, toll],
  );
  const { data, error, busy } = useSolve<RoutingSolution>("routing", body, 260);

  const sweepBodies = useMemo(
    () =>
      tollLink === null
        ? []
        : TOLL_POINTS.map((t) => ({ network: "sioux_falls", precision, max_od: 12, tolls: { [tollLink]: t } })),
    [tollLink, precision],
  );
  const swept = useSweep<RoutingSolution>("routing", sweepBodies, 700);

  const rows =
    tollLink !== null && swept.length === TOLL_POINTS.length
      ? TOLL_POINTS.map((t, i) => {
          const s = swept[i];
          return {
            x: num(t, 0),
            y: s ? count(s.total_cost) : "—",
            value: s && s.toll_effect ? money0(s.toll_effect.revenue) : "0",
            changed: !!s && !!swept[0] && s.total_cost < swept[0].total_cost - 1e-6,
          };
        })
      : [];

  const effect = data?.toll_effect ?? null;
  const busiest =
    data && links.length
      ? data.flows
          .map((f, i) => ({ i, vc: f / Math.max(links[i]?.capacity ?? 1, 1) }))
          .sort((a, b) => b.vc - a.vc)[0]
      : null;

  const maxFlow = data ? Math.max(...data.flows, 1) : 1;
  const named = (i: number | null) => (i === null || !links[i] ? "—" : `${links[i].from} → ${links[i].to}`);

  const headline = !data
    ? "Solving…"
    : tollLink === null
      ? `Total travel time ${count(data.total_cost)} vehicle-minutes with no toll.`
      : effect && effect.delta_total_cost < 0
        ? `A ${num(toll, 0)}-minute toll on ${named(tollLink)} saves the network ${count(-effect.delta_total_cost)} vehicle-minutes.`
        : effect
          ? `A ${num(toll, 0)}-minute toll on ${named(tollLink)} costs the network ${count(effect.delta_total_cost)} extra vehicle-minutes.`
          : `Total travel time ${count(data.total_cost)} vehicle-minutes.`;

  return (
    <div className="studio">
      <Answer headline={headline} busy={busy} error={error}>
        {data && (
          <>
            <Figure label="Total travel time" value={count(data.total_cost)} note="vehicle-minutes across the network" />
            <Figure label="Mean travel time" value={`${num(data.mean_travel_time, 2)} min`} note="per trip" tone="neutral" />
            <Figure
              label={effect ? "Toll revenue" : "Busiest link"}
              value={effect ? money0(effect.revenue) : busiest && links[busiest.i] ? named(busiest.i) : "—"}
              note={effect ? "toll times flow on the tolled link" : busiest ? `${(100 * busiest.vc).toFixed(0)}% of capacity` : ""}
              tone={effect && effect.delta_total_cost > 0 ? "warn" : "neutral"}
            />
          </>
        )}
      </Answer>

      {data && (
        <ModelLine>
          Stochastic user equilibrium, {data.n_links} links, {data.n_od} origin-destination pairs,{" "}
          {data.n_routes} routes, {count(data.total_demand)} trips, precision {num(data.precision, 2)}. Residual{" "}
          {data.residual.toExponential(1)} in {data.n_iter} iterations.
        </ModelLine>
      )}

      <div className="route-cols">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="route-map"
          style={{ opacity: busy ? 0.7 : 1 }}
          role="img"
          aria-label="Sioux Falls road network. Line width is link flow; colour is flow as a share of capacity. Use the dropdown below to toll a link with the keyboard."
        >
          {links.map((l, i) => {
            const [x1, y1] = toXY(...NODES[l.from - 1]);
            const [x2, y2] = toXY(...NODES[l.to - 1]);
            const flow = data ? data.flows[i] : 0;
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
          <Controls>
            <div className="knob">
              <span className="knob-label">
                <label htmlFor="toll-link">Link to toll</label>
                <output aria-hidden="true">{named(tollLink)}</output>
              </span>
              <select
                id="toll-link"
                value={tollLink === null ? "" : String(tollLink)}
                onChange={(e) => setTollLink(e.target.value === "" ? null : Number(e.target.value))}
              >
                <option value="">No toll</option>
                {links.map((l, i) => (
                  <option key={i} value={i}>
                    {l.from} → {l.to}
                  </option>
                ))}
              </select>
            </div>
            <Field
              label="Toll"
              value={toll}
              onChange={setToll}
              min={0}
              max={60}
              step={1}
              format={(x) => `${num(x, 0)} min`}
              help="In minutes of travel time drivers would pay to avoid the link."
            />
            <Field
              label="Route-choice precision"
              value={precision}
              onChange={setPrecision}
              min={0.05}
              max={3}
              step={0.05}
              format={(x) => num(x, 2)}
              help="Low spreads traffic over every plausible route; high piles it onto the fastest."
            />
          </Controls>

          {data && (
            <section className="card">
              <h3>Busiest links</h3>
              <table className="alt-table">
                <thead>
                  <tr>
                    <th scope="col">Link</th>
                    <th scope="col">Flow</th>
                    <th scope="col">Of capacity</th>
                  </tr>
                </thead>
                <tbody>
                  {data.flows
                    .map((f, i) => ({ i, f, vc: f / Math.max(links[i]?.capacity ?? 1, 1) }))
                    .sort((a, b) => b.vc - a.vc)
                    .slice(0, 5)
                    .map((r) => (
                      <tr key={r.i} data-us={r.i === tollLink}>
                        <th scope="row">{named(r.i)}</th>
                        <td>{count(r.f)}</td>
                        <td className={r.vc > 0.9 ? "alt-cost" : undefined}>{(100 * r.vc).toFixed(0)}%</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </section>
          )}
        </div>
      </div>

      {tollLink !== null && (
        <Sweep
          inputLabel="Toll (minutes)"
          choices={[{ key: "toll", label: `Toll on ${named(tollLink)}` }]}
          chosen="toll"
          onChoose={() => undefined}
          rows={rows}
          outputLabel="Total travel time"
          valueLabel="Toll revenue"
          note="Every row is a separate solve with only the toll changed. Rows that lower total travel time against no toll are marked."
        />
      )}
    </div>
  );
}
