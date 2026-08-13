"use client";

/* The merit order and where it clears.
 *
 * Generators are stacked left to right in offer order; the width of a block
 * is its capacity, the height its offer. Demand is the vertical rule, and
 * the clearing price is read off the block demand lands in — the cross.
 * Block heights tween, so raising an offer visibly lifts the stack and, if
 * it lifts far enough, moves the cross.
 */

import { useTweenArray } from "../../../lib/anim";
import { extent, linear, TableTwin } from "./primitives";

const W = 640;
const H = 250;
const PAD = { l: 58, r: 18, t: 16, b: 38 };

export interface Unit {
  label: string;
  offer: number;
  capacity: number;
  cost: number;
  mine: boolean;
}

export function OfferStack({
  units,
  demand,
  clearingPrice,
  formatPrice,
  formatMW,
}: {
  units: Unit[];
  demand: number;
  clearingPrice: number;
  formatPrice: (v: number) => string;
  formatMW: (v: number) => string;
}) {
  const order = [...units].sort((a, b) => a.offer - b.offer);
  const offers = useTweenArray(order.map((u) => u.offer), 380);
  const capTotal = order.reduce((s, u) => s + u.capacity, 0);

  const xs = linear([0, Math.max(capTotal, demand * 1.15)], [PAD.l, W - PAD.r]);
  const [lo, hi] = extent([0, ...order.map((u) => u.offer), clearingPrice], true);
  const ys = linear([Math.min(0, lo), hi], [H - PAD.b, PAD.t]);

  let cursor = 0;
  const blocks = order.map((u, i) => {
    const x0 = cursor;
    cursor += u.capacity;
    return { u, x0, x1: cursor, offer: offers[i] ?? u.offer };
  });

  const priceTicks = ys.ticks(4);
  const clearX = xs(Math.min(demand, capTotal));

  return (
    <div>
      <div className="chart-legend">
        <span>
          <i className="legend-key" style={{ background: "var(--series-1)" }} />
          your unit
        </span>
        <span>
          <i className="legend-key" style={{ background: "var(--series-3)" }} />
          the other unit
        </span>
        <span>
          <i className="legend-key" style={{ background: "var(--text-2)" }} />
          demand
        </span>
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="chart chart-frame"
        role="img"
        aria-label={`Offer stack. Demand of ${formatMW(demand)} clears at ${formatPrice(
          clearingPrice,
        )}. The table below carries every block.`}
      >
        {priceTicks.map((t) => (
          <g key={t} aria-hidden>
            <line x1={PAD.l} x2={W - PAD.r} y1={ys(t)} y2={ys(t)} stroke="var(--border)" />
            <text x={PAD.l - 8} y={ys(t) + 3.5} textAnchor="end" fontSize="10.5" fill="var(--text-3)" fontFamily="var(--mono)">
              {formatPrice(t)}
            </text>
          </g>
        ))}

        {blocks.map((b, i) => {
          const yTop = ys(Math.max(b.offer, 0));
          const yBase = ys(0);
          const x0 = xs(b.x0) + 1;
          const x1 = xs(b.x1) - 1;
          return (
            <g key={i}>
              {/* A supply block is a large area, so it is a wash, not a slab: the
                  offer is carried by the 2.5px step across the top. */}
              <rect
                x={x0}
                y={Math.min(yTop, yBase)}
                width={Math.max(2, x1 - x0)}
                height={Math.max(1.5, Math.abs(yBase - yTop))}
                rx="3"
                fill={b.u.mine ? "var(--series-1)" : "var(--series-3)"}
                opacity="0.14"
              />
              <path
                d={`M${x0},${yBase} L${x0},${yTop} L${x1},${yTop} L${x1},${yBase}`}
                fill="none"
                stroke={b.u.mine ? "var(--series-1)" : "var(--series-3)"}
                strokeWidth="2.5"
                strokeLinejoin="round"
              />
              <line
                x1={x0}
                x2={x1}
                y1={ys(b.u.cost)}
                y2={ys(b.u.cost)}
                stroke="var(--text-2)"
                strokeWidth="1.25"
                strokeDasharray="4 4"
              />
              <text
                x={(x0 + x1) / 2}
                y={H - PAD.b + 15}
                textAnchor="middle"
                fontSize="10.5"
                fill="var(--text-3)"
                fontFamily="var(--mono)"
              >
                {b.u.label}
              </text>
            </g>
          );
        })}

        <line x1={PAD.l} x2={W - PAD.r} y1={ys(0)} y2={ys(0)} stroke="var(--border-strong)" />

        <line x1={clearX} y1={PAD.t} x2={clearX} y2={H - PAD.b} stroke="var(--text-2)" strokeWidth="1.5" />
        <line
          x1={PAD.l}
          x2={W - PAD.r}
          y1={ys(clearingPrice)}
          y2={ys(clearingPrice)}
          stroke="var(--text-2)"
          strokeWidth="1.5"
          strokeDasharray="6 4"
        />
        <circle cx={clearX} cy={ys(clearingPrice)} r="6.5" fill="var(--surface)" />
        <circle cx={clearX} cy={ys(clearingPrice)} r="4.5" fill="var(--text)" />
        <text
          x={Math.min(clearX + 9, W - PAD.r - 118)}
          y={Math.max(PAD.t + 11, ys(clearingPrice) - 9)}
          fontSize="11"
          fontWeight="600"
          fill="var(--text)"
          fontFamily="var(--mono)"
        >
          clears {formatPrice(clearingPrice)}
        </text>
        <text x={(W + PAD.l) / 2} y={H - 3} textAnchor="middle" fontSize="10.5" fill="var(--text-3)">
          capacity offered, cumulative
        </text>
      </svg>
      <p className="chart-note">
        Dashed rule inside a block is that unit&apos;s marginal cost; the gap above it is the mark-up it is offering.
      </p>
      <TableTwin
        caption="The stack, block by block"
        columns={["Unit", "Offer", "Cost", "Capacity"]}
        rows={order.map((u) => [u.label, formatPrice(u.offer), formatPrice(u.cost), formatMW(u.capacity)])}
      />
    </div>
  );
}
