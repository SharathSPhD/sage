"use client";

/* The Sioux Falls network, drawn from its published node coordinates.
 *
 * Line width is the flow the solver returned; the marching dashes along a
 * link move at a speed set by that link's flow, so congestion reads as
 * motion as well as thickness. Colour is how close the link is to capacity,
 * always paired with a word in the legend and the table, never alone.
 * Every link is a button: clickable with a pointer, reachable with Tab.
 */

import { useState } from "react";
import { useTweenArray, usePrefersReducedMotion } from "../../../lib/anim";
import { TableTwin } from "./primitives";

// TNTP SiouxFalls_node.tntp coordinates (lon, lat) — the drawing layout only;
// every flow on screen comes back from the solver.
export const SIOUX_NODES: [number, number][] = [
  [-96.7704, 43.6128], [-96.7113, 43.6058], [-96.7743, 43.573], [-96.7472, 43.5637],
  [-96.7316, 43.564], [-96.7116, 43.5876], [-96.6934, 43.5638], [-96.7114, 43.5623],
  [-96.7312, 43.5486], [-96.7314, 43.5453], [-96.7468, 43.5441], [-96.7801, 43.5439],
  [-96.7934, 43.4907], [-96.751, 43.5293], [-96.7315, 43.5294], [-96.7114, 43.5467],
  [-96.7114, 43.5413], [-96.6941, 43.5467], [-96.7113, 43.5296], [-96.7112, 43.5153],
  [-96.731, 43.5105], [-96.7312, 43.5149], [-96.7509, 43.5149], [-96.7492, 43.5032],
];

export interface NetLink {
  from: number;
  to: number;
  capacity: number;
}

const W = 520;
const H = 620;

function project(lon: number, lat: number): [number, number] {
  const lons = SIOUX_NODES.map((n) => n[0]);
  const lats = SIOUX_NODES.map((n) => n[1]);
  const lo0 = Math.min(...lons);
  const lo1 = Math.max(...lons);
  const la0 = Math.min(...lats);
  const la1 = Math.max(...lats);
  const pad = 38;
  return [
    pad + ((lon - lo0) / (lo1 - lo0)) * (W - 2 * pad),
    pad + ((la1 - lat) / (la1 - la0)) * (H - 2 * pad),
  ];
}

/** Under half capacity, near it, over it — a word for every band. */
export function loadBand(vc: number): { key: "clear" | "busy" | "over"; label: string; color: string } {
  if (vc > 0.9) return { key: "over", label: "over capacity", color: "var(--danger)" };
  if (vc > 0.55) return { key: "busy", label: "near capacity", color: "var(--warn)" };
  return { key: "clear", label: "free flowing", color: "var(--series-1)" };
}

export function NetworkMap({
  links,
  flows,
  travelTimes,
  tolled,
  onToll,
  busy,
}: {
  links: NetLink[];
  flows: number[];
  travelTimes?: number[];
  tolled: number | null;
  onToll: (index: number | null) => void;
  busy?: boolean;
}) {
  const reduced = usePrefersReducedMotion();
  // WCAG 2.2.2 — the marching flow starts on its own and never stops, so it
  // needs a control that stops it. Reduced motion suppresses it outright.
  const [flowing, setFlowing] = useState(true);
  const animate = flowing && !reduced;
  const smooth = useTweenArray(flows, 500);
  const max = Math.max(...smooth, 1);

  return (
    <div>
      <div className="chart-legend">
        <span>
          <i className="legend-key" style={{ background: "var(--series-1)" }} />
          free flowing
        </span>
        <span>
          <i className="legend-key" style={{ background: "var(--warn)" }} />
          near capacity
        </span>
        <span>
          <i className="legend-key" style={{ background: "var(--danger)" }} />
          over capacity
        </span>
        <span>
          <i className="legend-key" style={{ background: "var(--text)" }} />
          charged
        </span>
        {!reduced && (
          <button
            type="button"
            data-quiet="true"
            style={{ marginLeft: "auto", fontSize: "var(--text-sm)", minHeight: "32px" }}
            aria-pressed={animate}
            onClick={() => setFlowing((f) => !f)}
          >
            {animate ? "Stop the flow animation" : "Animate the flow"}
          </button>
        )}
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="route-map"
        style={{ opacity: busy ? 0.72 : 1, transition: "opacity 150ms" }}
        role="group"
        aria-label="Sioux Falls road network. Line width is traffic volume, colour is how close each road is to capacity. Each road is a button that charges for it."
      >
        {links.map((l, i) => {
          const a = SIOUX_NODES[l.from - 1];
          const b = SIOUX_NODES[l.to - 1];
          if (!a || !b) return null;
          const [x1, y1] = project(a[0], a[1]);
          const [x2, y2] = project(b[0], b[1]);
          const flow = smooth[i] ?? 0;
          const band = loadBand(flow / Math.max(l.capacity, 1));
          const isTolled = tolled === i;
          const width = 0.9 + 5.4 * (flow / max);
          return (
            <g key={i}>
              <line
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke={isTolled ? "var(--text)" : band.color}
                strokeWidth={width}
                strokeLinecap="round"
                opacity={isTolled ? 1 : 0.85}
              />
              {animate && flow / max > 0.18 && (
                <line
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke="var(--surface)"
                  strokeWidth={Math.max(0.8, width * 0.45)}
                  strokeLinecap="round"
                  opacity="0.55"
                  className="flow-dash"
                  style={{ animationDuration: `${Math.max(0.45, 2.4 - 2 * (flow / max))}s` }}
                />
              )}
              <line
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke="transparent"
                strokeWidth={Math.max(14, width + 10)}
                className="route-link"
                role="button"
                tabIndex={0}
                aria-pressed={isTolled}
                aria-label={`Road ${l.from} to ${l.to}, ${Math.round(flow).toLocaleString()} vehicles, ${band.label}. ${
                  isTolled ? "Charged. Activate to remove the charge." : "Activate to charge for it."
                }`}
                onClick={() => onToll(isTolled ? null : i)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onToll(isTolled ? null : i);
                  }
                }}
              />
            </g>
          );
        })}
        {SIOUX_NODES.map(([lon, lat], k) => {
          const [x, y] = project(lon, lat);
          return (
            <g key={k} aria-hidden>
              <circle cx={x} cy={y} r="8" fill="var(--surface)" stroke="var(--border-strong)" strokeWidth="1" />
              <text x={x} y={y + 3.2} textAnchor="middle" fontSize="8" fill="var(--text-2)" fontFamily="var(--mono)">
                {k + 1}
              </text>
            </g>
          );
        })}
      </svg>
      <TableTwin
        caption="Every road, with its flow"
        columns={["Road", "Vehicles", "State", travelTimes ? "Minutes" : "Capacity"]}
        rows={links
          .map((l, i) => ({ l, i, f: flows[i] ?? 0 }))
          .sort((a, b) => b.f - a.f)
          .map(({ l, i, f }) => [
            `${l.from} → ${l.to}`,
            Math.round(f).toLocaleString(),
            loadBand(f / Math.max(l.capacity, 1)).label,
            travelTimes ? (travelTimes[i] ?? 0).toFixed(1) : Math.round(l.capacity).toLocaleString(),
          ])}
      />
    </div>
  );
}
