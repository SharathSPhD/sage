"use client";

/* A value curve over a decision grid, with the recommended level called out.
 *
 * The curve morphs between solutions rather than redrawing: the y-values
 * are tweened on requestAnimationFrame and the path is rebuilt each frame,
 * so the shape flows from one solve to the next. On first mount it traces
 * itself once. Both stop dead under prefers-reduced-motion.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useTweenArray, usePrefersReducedMotion } from "../../../lib/anim";
import { AxisX, Grid, PAD, areaOf, extent, linear, pathOf, useNearest, TableTwin } from "./primitives";

const W = 640;
const H = 230;

export interface CurveChartProps {
  x: number[];
  y: number[];
  markIndex: number;
  formatX: (v: number) => string;
  formatY: (v: number) => string;
  xLabel: string;
  yLabel: string;
  /** Optional second curve, plotted on the SAME scale (never a second axis). */
  compare?: { values: number[]; label: string; color?: string };
  seriesLabel?: string;
  markLabel?: string;
  tableCaption?: string;
}

export function CurveChart({
  x,
  y,
  markIndex,
  formatX,
  formatY,
  xLabel,
  yLabel,
  compare,
  seriesLabel,
  markLabel = "recommended",
  tableCaption,
}: CurveChartProps) {
  const reduced = usePrefersReducedMotion();
  const smooth = useTweenArray(y, 420);
  const smoothCompare = useTweenArray(compare?.values ?? [], 420);
  const [traced, setTraced] = useState(false);
  const lengthRef = useRef(1400);

  useEffect(() => {
    const t = setTimeout(() => setTraced(true), 950);
    return () => clearTimeout(t);
  }, []);

  const all = compare ? [...y, ...compare.values] : y;
  const [lo, hi] = extent(all, true);
  const xs = linear([0, Math.max(1, x.length - 1)], [PAD.l, W - PAD.r]);
  const ys = linear([lo, hi], [H - PAD.b, PAD.t]);
  const at = (i: number) => i;

  const { ref, hover, onMove, clear } = useNearest(x.length, xs);

  const d = useMemo(() => pathOf(smooth, xs, ys, at), [smooth, xs, ys]);
  const area = useMemo(() => areaOf(smooth, xs, ys, at, Math.max(lo, 0)), [smooth, xs, ys, lo]);
  const dCompare = useMemo(
    () => (compare && smoothCompare.length === compare.values.length ? pathOf(smoothCompare, xs, ys, at) : ""),
    [compare, smoothCompare, xs, ys],
  );

  if (x.length < 2 || y.length < 2) return null;

  const yTicks = ys.ticks(4);
  const markX = markIndex >= 0 && markIndex < x.length ? xs(markIndex) : null;
  const markY = markIndex >= 0 && markIndex < smooth.length ? ys(smooth[markIndex]) : null;
  const hoverI = hover?.index ?? null;

  return (
    <div className="chart-host">
      {compare && (
        <div className="chart-legend">
          <span>
            <i className="legend-key" style={{ background: "var(--series-1)" }} />
            {seriesLabel ?? yLabel}
          </span>
          <span>
            <i className="legend-key" style={{ background: compare.color ?? "var(--series-2)" }} />
            {compare.label}
          </span>
        </div>
      )}
      <svg
        ref={ref}
        viewBox={`0 0 ${W} ${H}`}
        className="chart chart-frame"
        role="img"
        onPointerMove={onMove}
        onPointerLeave={clear}
        aria-label={`${yLabel} against ${xLabel}. Best at ${formatX(x[Math.max(0, markIndex)])}, ${formatY(
          y[Math.max(0, markIndex)] ?? 0,
        )}. The table under the chart carries every value.`}
      >
        <Grid y={ys} x0={PAD.l} x1={W - PAD.r} ticks={yTicks} format={formatY} />
        <AxisX
          x={xs}
          yPix={H - PAD.b}
          labels={[
            { at: 0, text: formatX(x[0]) },
            { at: (x.length - 1) / 2, text: formatX(x[Math.floor((x.length - 1) / 2)]) },
            { at: x.length - 1, text: formatX(x[x.length - 1]) },
          ]}
        />
        <text
          x={(W + PAD.l) / 2}
          y={H - 3}
          textAnchor="middle"
          fontSize="10.5"
          fill="var(--text-3)"
          fontFamily="var(--sans)"
        >
          {xLabel}
        </text>

        <path d={area} fill="var(--series-1)" opacity="0.1" />
        {dCompare && (
          <path
            d={dCompare}
            fill="none"
            stroke={compare?.color ?? "var(--series-2)"}
            strokeWidth="2"
            strokeLinejoin="round"
            strokeLinecap="round"
            strokeDasharray="5 4"
          />
        )}
        <path
          d={d}
          fill="none"
          stroke="var(--series-1)"
          strokeWidth="2"
          strokeLinejoin="round"
          strokeLinecap="round"
          className={!reduced && !traced ? "trace-path" : undefined}
          style={!reduced && !traced ? ({ ["--draw-length" as string]: lengthRef.current } as React.CSSProperties) : undefined}
        />

        {markX !== null && markY !== null && (
          <g>
            <line
              x1={markX}
              y1={PAD.t}
              x2={markX}
              y2={H - PAD.b}
              stroke="var(--border-strong)"
              strokeWidth="1"
            />
            {/* The callout sits on whichever side of the mark has room, on its own
                surface patch, so it never has to be read through the curve. */}
            {(() => {
              const width = Math.max(58, formatX(x[markIndex]).length * 7.4 + 16);
              const left = markX > W - PAD.r - width - 12;
              const bx = left ? markX - width - 8 : markX + 8;
              return (
                <g>
                  <rect x={bx} y={PAD.t - 2} width={width} height="30" rx="5" fill="var(--surface)" opacity="0.92" />
                  <text
                    x={bx + 8}
                    y={PAD.t + 11}
                    fontSize="11"
                    fill="var(--text)"
                    fontFamily="var(--mono)"
                    fontWeight="600"
                  >
                    {formatX(x[markIndex])}
                  </text>
                  <text x={bx + 8} y={PAD.t + 23} fontSize="9.5" fill="var(--text-3)" fontFamily="var(--sans)">
                    {markLabel}
                  </text>
                </g>
              );
            })()}
            <circle cx={markX} cy={markY} r="6.5" fill="var(--surface)" />
            <circle cx={markX} cy={markY} r="4.5" fill="var(--series-1)" />
          </g>
        )}

        {hoverI !== null && hoverI !== markIndex && (
          <g aria-hidden>
            <line x1={xs(hoverI)} y1={PAD.t} x2={xs(hoverI)} y2={H - PAD.b} stroke="var(--border-strong)" strokeDasharray="2 3" />
            <circle cx={xs(hoverI)} cy={ys(smooth[hoverI] ?? 0)} r="5.5" fill="var(--surface)" />
            <circle cx={xs(hoverI)} cy={ys(smooth[hoverI] ?? 0)} r="3.5" fill="var(--series-1)" />
          </g>
        )}
      </svg>

      {hover && hoverI !== null && (
        <div
          className="chart-tooltip"
          style={{
            left: Math.min(hover.left + 12, 520),
            top: Math.max(0, hover.top - 44),
          }}
        >
          <strong>{formatX(x[hoverI])}</strong> · {formatY(y[hoverI])}
          {compare ? ` · ${compare.label} ${formatY(compare.values[hoverI])}` : ""}
        </div>
      )}

      <TableTwin
        caption={tableCaption ?? "Every point on this chart, as a table"}
        columns={compare ? [xLabel, yLabel, compare.label] : [xLabel, yLabel]}
        rows={x.map((v, i) => (compare ? [formatX(v), formatY(y[i]), formatY(compare.values[i])] : [formatX(v), formatY(y[i])]))}
      />
    </div>
  );
}
