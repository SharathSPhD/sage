"use client";

/* Chart primitives shared by every visual on the site.
 *
 * House rules, applied here once so no chart has to remember them:
 *   - one axis, never two;
 *   - gridlines and axes are solid hairlines one step off the surface;
 *   - marks carry the series colour, text never does;
 *   - a legend whenever there are two or more series, and labels are
 *     placed selectively rather than on every point;
 *   - every chart has a table twin so no value is reachable only by hover.
 */

import { useCallback, useMemo, useRef, useState } from "react";

export interface Scale {
  (v: number): number;
  domain: [number, number];
  range: [number, number];
  ticks: (n: number) => number[];
  invert: (px: number) => number;
}

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1e-12;
  const fn = ((v: number) => r0 + ((v - d0) / span) * (r1 - r0)) as Scale;
  fn.domain = domain;
  fn.range = range;
  fn.invert = (px: number) => d0 + ((px - r0) / (r1 - r0 || 1e-12)) * span;
  fn.ticks = (n: number) => niceTicks(d0, d1, n);
  return fn;
}

/** Round tick values, the ones a reader would have chosen. */
export function niceTicks(lo: number, hi: number, count = 4): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  // 1 / 2 / 2.5 / 5 / 10 — 2.5 is what keeps a 0-to-80 axis from collapsing to
  // two ticks, which is the usual way a "nice" ladder goes wrong.
  const step = (norm > 5 ? 10 : norm > 2.5 ? 5 : norm > 2 ? 2.5 : norm > 1 ? 2 : 1) * mag;
  const out: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) {
    out.push(Number(t.toPrecision(12)));
  }
  return out;
}

export const PAD = { l: 58, r: 18, t: 14, b: 34 };

export function Grid({
  y,
  x0,
  x1,
  ticks,
  format,
}: {
  y: Scale;
  x0: number;
  x1: number;
  ticks: number[];
  format: (v: number) => string;
}) {
  return (
    <g aria-hidden>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x0} x2={x1} y1={y(t)} y2={y(t)} stroke="var(--border)" strokeWidth="1" />
          <text
            x={x0 - 8}
            y={y(t) + 3.5}
            textAnchor="end"
            fontSize="10.5"
            fill="var(--text-3)"
            fontFamily="var(--mono)"
          >
            {format(t)}
          </text>
        </g>
      ))}
    </g>
  );
}

export function AxisX({
  x,
  yPix,
  labels,
}: {
  x: Scale;
  yPix: number;
  labels: { at: number; text: string }[];
}) {
  return (
    <g aria-hidden>
      <line x1={x.range[0]} x2={x.range[1]} y1={yPix} y2={yPix} stroke="var(--border-strong)" strokeWidth="1" />
      {labels.map((l, i) => (
        <text
          key={i}
          x={x(l.at)}
          y={yPix + 16}
          textAnchor={i === 0 ? "start" : i === labels.length - 1 ? "end" : "middle"}
          fontSize="10.5"
          fill="var(--text-3)"
          fontFamily="var(--mono)"
        >
          {l.text}
        </text>
      ))}
    </g>
  );
}

export function Legend({ items }: { items: { label: string; color: string; shape?: "line" | "dot" }[] }) {
  if (items.length < 2) return null;
  return (
    <div className="chart-legend">
      {items.map((it) => (
        <span key={it.label}>
          <i className="legend-key" data-shape={it.shape === "dot" ? "dot" : "line"} style={{ background: it.color }} />
          {it.label}
        </span>
      ))}
    </div>
  );
}

/** Every chart's WCAG-clean twin: the same numbers, as a table. */
export function TableTwin({
  caption,
  columns,
  rows,
  open = false,
}: {
  caption: string;
  columns: string[];
  rows: (string | number)[][];
  open?: boolean;
}) {
  return (
    <details className="under-the-hood" open={open}>
      <summary>{caption}</summary>
      <div style={{ maxHeight: "18rem", overflow: "auto", marginBottom: "0.8rem" }}>
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c} scope="col">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                {r.map((cell, j) =>
                  j === 0 ? (
                    <th key={j} scope="row">
                      {cell}
                    </th>
                  ) : (
                    <td key={j}>{cell}</td>
                  ),
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

export interface HoverState {
  index: number;
  left: number;
  top: number;
}

/**
 * Pointer tracking over a plot area, in viewBox units, snapped to the
 * nearest sample. The hit area is the whole plot, so no reader has to land
 * on a mark.
 */
export function useNearest(count: number, x: Scale) {
  const ref = useRef<SVGSVGElement | null>(null);
  const [hover, setHover] = useState<HoverState | null>(null);

  const onMove = useCallback(
    (e: React.PointerEvent<SVGSVGElement>) => {
      const svg = ref.current;
      if (!svg || count < 2) return;
      const box = svg.getBoundingClientRect();
      const vb = svg.viewBox.baseVal;
      const px = ((e.clientX - box.left) / box.width) * vb.width;
      const u = (px - x.range[0]) / (x.range[1] - x.range[0] || 1);
      const index = Math.max(0, Math.min(count - 1, Math.round(u * (count - 1))));
      setHover({ index, left: e.clientX - box.left, top: e.clientY - box.top });
    },
    [count, x],
  );

  const clear = useCallback(() => setHover(null), []);
  return { ref, hover, onMove, clear, setHover };
}

/** A path through (i, v) samples, as an SVG `d`. */
export function pathOf(values: number[], x: Scale, y: Scale, xAt: (i: number) => number): string {
  return values
    .map((v, i) => `${i === 0 ? "M" : "L"}${x(xAt(i)).toFixed(2)},${y(v).toFixed(2)}`)
    .join(" ");
}

/** The same path closed to a baseline, for the 10% wash under a line. */
export function areaOf(values: number[], x: Scale, y: Scale, xAt: (i: number) => number, base: number): string {
  if (values.length < 2) return "";
  const top = pathOf(values, x, y, xAt);
  return `${top} L${x(xAt(values.length - 1)).toFixed(2)},${y(base).toFixed(2)} L${x(xAt(0)).toFixed(2)},${y(base).toFixed(2)} Z`;
}

/** Extent with a little headroom, never inverted. */
export function extent(values: number[], includeZero = false): [number, number] {
  const vals = values.filter((v) => Number.isFinite(v));
  if (vals.length === 0) return [0, 1];
  let lo = Math.min(...vals);
  let hi = Math.max(...vals);
  if (includeZero) lo = Math.min(lo, 0);
  if (hi === lo) {
    hi = lo + Math.abs(lo || 1) * 0.1;
  }
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

export function useIds(n: number, prefix: string): string[] {
  return useMemo(() => Array.from({ length: n }, (_, i) => `${prefix}-${i}`), [n, prefix]);
}
