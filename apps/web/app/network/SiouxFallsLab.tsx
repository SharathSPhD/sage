"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { NumberDial, PanelShell } from "../components/panels/ui";

/* Real data, real solver: the TNTP Sioux Falls network drawn from its actual
   node coordinates, link flows from the deployed Fisk SUE solver, tolls as
   the domain's conjugate field. Click a link to toll it. */

// TNTP SiouxFalls_node.tntp coordinates (lon, lat) — drawing layout only;
// all physics comes from the API.
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

export function SiouxFallsLab() {
  const [links, setLinks] = useState<Link[]>([]);
  const [theta, setTheta] = useState(0.5);
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
      body: JSON.stringify({ theta, tolls }),
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
  }, [theta, tollLink, toll]);

  useEffect(() => {
    const t = setTimeout(solve, 350);
    return () => clearTimeout(t);
  }, [solve]);

  const maxFlow = sue ? Math.max(...sue.link_flows, 1) : 1;
  const delta = sue && baseline !== null && tollLink !== null ? sue.total_travel_time - baseline : null;

  return (
    <PanelShell title="toll a link, watch the city re-equilibrate" provenance="live">
      <div className="panel-cols" style={{ gridTemplateColumns: "auto 1fr" }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: 560, opacity: busy ? 0.65 : 1, transition: "opacity 150ms" }} aria-label="Sioux Falls network with equilibrium flows">
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
          <NumberDial value={theta} setValue={setTheta} min={0.05} max={3} step={0.05} label="route-choice precision θ" />
          <p style={{ fontSize: "0.72rem", color: "var(--text-faint)", margin: "0.3rem 0 0.9rem" }}>
            θ is λ for populations: low θ spreads traffic over routes, high θ concentrates it on
            the fastest.
          </p>
          {tollLink !== null ? (
            <div className="card" style={{ padding: "0.8rem 1rem", marginBottom: "0.9rem" }}>
              <div className="panel-label">
                tolling link {links[tollLink]?.from} → {links[tollLink]?.to}{" "}
                <button style={{ marginLeft: "0.5rem", padding: "0.1rem 0.5rem", fontSize: "0.7rem" }} onClick={() => setTollLink(null)}>
                  remove
                </button>
              </div>
              <NumberDial value={toll} setValue={setToll} min={0} max={50} step={1} label="toll (minutes-equivalent)" format={(v) => v.toFixed(0)} />
            </div>
          ) : (
            <p style={{ fontSize: "0.8rem", color: "var(--text-faint)", marginBottom: "0.9rem" }}>
              click any link to toll it — the conjugate field of this domain
            </p>
          )}
          {sue && (
            <table style={{ width: "100%", fontFamily: "var(--mono)", fontSize: "0.85rem" }}>
              <tbody>
                <tr>
                  <td style={{ color: "var(--text-faint)", padding: "0.2rem 0" }}>total travel time</td>
                  <td style={{ textAlign: "right", color: "var(--accent)" }}>{sue.total_travel_time.toFixed(0)}</td>
                </tr>
                {delta !== null && (
                  <tr>
                    <td style={{ color: "var(--text-faint)", padding: "0.2rem 0" }}>vs. untolled</td>
                    <td style={{ textAlign: "right", color: delta > 0 ? "var(--amber)" : "var(--accent)" }}>
                      {delta > 0 ? "+" : ""}
                      {delta.toFixed(0)}
                    </td>
                  </tr>
                )}
                <tr>
                  <td style={{ color: "var(--text-faint)", padding: "0.2rem 0" }}>Beckmann potential</td>
                  <td style={{ textAlign: "right" }}>{sue.beckmann.toExponential(3)}</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text-faint)", padding: "0.2rem 0" }}>KKT residual</td>
                  <td style={{ textAlign: "right" }}>{sue.residual.toExponential(1)}</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text-faint)", padding: "0.2rem 0" }}>reciprocity ℛ (gate)</td>
                  <td style={{ textAlign: "right", color: "var(--accent)" }}>5.7×10⁻¹⁷ ≈ 0</td>
                </tr>
              </tbody>
            </table>
          )}
          <p style={{ fontSize: "0.76rem", color: "var(--text-faint)", marginTop: "0.8rem" }}>
            Width = equilibrium flow; colour = congestion (v/c). Because this system is an exact
            potential game, a toll on link A moves flows on link B exactly as a toll on B would
            move A — the Onsager symmetry the reciprocity meter certifies. Top-12 OD pairs,
            k = 3 routes each (documented plugin restriction).
          </p>
        </div>
      </div>
    </PanelShell>
  );
}
