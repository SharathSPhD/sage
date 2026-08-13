"use client";

/* The price ladder: every level on the grid as a rung, carrying what it
 * earns you and how likely the rival is to sit there. The recommended rung
 * is the only one labelled. Bars tween, so a change of cost walks the
 * profit down the ladder rather than repainting it.
 */

import { useTweenArray } from "../../../lib/anim";
import { TableTwin } from "./primitives";

export function PriceLadder({
  levels,
  profit,
  rival,
  chosen,
  formatLevel,
  formatProfit,
  levelLabel = "Price",
  valueLabel = "Expected profit",
  rivalLabel = "Rival sits here",
  limit = 14,
}: {
  levels: number[];
  profit: number[];
  rival: number[];
  chosen: number;
  formatLevel: (v: number) => string;
  formatProfit: (v: number) => string;
  levelLabel?: string;
  valueLabel?: string;
  rivalLabel?: string;
  limit?: number;
}) {
  const keep = pickWindow(levels.length, chosen, limit);
  const p = useTweenArray(keep.map((i) => profit[i] ?? 0), 420);
  const r = useTweenArray(keep.map((i) => rival[i] ?? 0), 420);

  const lo = Math.min(...p, 0);
  const hi = Math.max(...p, 1e-9);
  const rMax = Math.max(...r, 1e-9);

  return (
    <div>
      <div className="chart-legend">
        <span>
          <i className="legend-key" style={{ background: "var(--series-1)" }} />
          {valueLabel}
        </span>
        <span>
          <i className="legend-key" style={{ background: "var(--series-2)" }} />
          {rivalLabel}
        </span>
      </div>
      <ul className="rival-bars" style={{ gap: "0.25rem" }}>
        {keep.map((idx, k) => {
          const isPick = idx === chosen;
          const w = (100 * (p[k] - lo)) / (hi - lo || 1);
          const rw = (100 * r[k]) / rMax;
          return (
            <li key={idx} style={{ alignItems: "stretch" }}>
              <span
                className="rb-label"
                style={{
                  width: "5.2rem",
                  color: isPick ? "var(--text)" : "var(--text-2)",
                  fontWeight: isPick ? 700 : 400,
                  alignSelf: "center",
                }}
              >
                {formatLevel(levels[idx])}
              </span>
              <span style={{ flex: 1, display: "flex", flexDirection: "column", gap: "2px", minWidth: 0 }}>
                <span className="rb-track" style={{ height: "0.85rem" }}>
                  <span className="rb-fill" style={{ width: `${Math.max(0.8, w)}%`, transition: "none" }} />
                </span>
                <span className="rb-rail" style={{ height: "0.4rem", display: "block" }}>
                  <span
                    className="rb-fill"
                    data-tone="rival"
                    style={{ display: "block", height: "100%", width: `${Math.max(0.8, rw)}%`, transition: "none" }}
                  />
                </span>
              </span>
              <span className="rb-val" style={{ width: "6.4rem", alignSelf: "center" }}>
                {isPick ? <strong>{formatProfit(profit[idx])}</strong> : formatProfit(profit[idx])}
              </span>
            </li>
          );
        })}
      </ul>
      <p className="chart-note">
        The rung in bold is the recommendation. The thin bar under each rung is how much of the rival&apos;s own
        distribution sits at that level.
      </p>
      <TableTwin
        caption="Every level on the grid"
        columns={[levelLabel, valueLabel, rivalLabel]}
        rows={levels.map((v, i) => [formatLevel(v), formatProfit(profit[i] ?? 0), `${(100 * (rival[i] ?? 0)).toFixed(1)}%`])}
      />
    </div>
  );
}

/** A window of rungs around the chosen one, so a 60-level grid stays readable. */
function pickWindow(n: number, chosen: number, limit: number): number[] {
  if (n <= limit) return Array.from({ length: n }, (_, i) => i);
  const half = Math.floor(limit / 2);
  let start = Math.max(0, Math.min(n - limit, (chosen < 0 ? 0 : chosen) - half));
  if (start < 0) start = 0;
  return Array.from({ length: limit }, (_, i) => start + i);
}
