"use client";

/* Traffic assignment on the Sioux Falls benchmark, with a toll on one link.
 *
 * POST /v1/solve/routing — body mirrors sq.RoutingProblem(network=, tolls=,
 * precision=, max_od=). Link flows, travel times, total cost and the toll
 * effect all come back from that call; the map only supplies coordinates.
 */

import { useEffect, useMemo, useState } from "react";
import { useSolve, useSweep, type RoutingSolution } from "../../../lib/problems";
import { NetworkMap } from "../charts/NetworkMap";
import { Answer, Controls, Field, Figure, ModelLine, Sweep, count, money0, num } from "./ui";

interface Link {
  from: number;
  to: number;
  free_flow: number;
  capacity: number;
}

const TOLL_POINTS = [0, 5, 10, 20, 35, 50];

export function RoutingSolver() {
  const [links, setLinks] = useState<Link[]>([]);
  const [precision, setPrecision] = useState(0.5);
  const [tollLink, setTollLink] = useState<number | null>(null);
  const [toll, setToll] = useState(10);

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

  const named = (i: number | null) => (i === null || !links[i] ? "—" : `${links[i].from} → ${links[i].to}`);

  const headline = !data
    ? "Solving…"
    : tollLink === null && busiest && links[busiest.i]
      ? `Start with ${named(busiest.i)} — it is running at ${(100 * busiest.vc).toFixed(0)}% of capacity. Click any road on the map to price it.`
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
        <NetworkMap
          links={links}
          flows={data ? data.flows : links.map(() => 0)}
          travelTimes={data ? data.travel_times : undefined}
          tolled={tollLink}
          onToll={setTollLink}
          busy={busy}
        />

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
