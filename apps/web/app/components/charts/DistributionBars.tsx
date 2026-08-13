"use client";

/* A distribution over labelled levels. Bar widths are tweened, so moving
 * the precision slider makes the mass visibly gather or spread instead of
 * jumping between two static pictures.
 */

import { useTweenArray } from "../../../lib/anim";

export interface DistRow {
  label: string;
  p: number;
}

export function DistributionBars({
  rows,
  limit = 10,
  tone = "own",
  digits,
}: {
  rows: DistRow[];
  limit?: number;
  tone?: "own" | "rival";
  digits?: number;
}) {
  const shown = rows.length > limit ? [...rows].sort((a, b) => b.p - a.p).slice(0, limit) : rows;
  const smooth = useTweenArray(shown.map((r) => r.p), 380);
  const max = Math.max(...smooth, 1e-9);

  return (
    <>
      <ul className="rival-bars">
        {shown.map((r, i) => {
          const p = smooth[i] ?? r.p;
          return (
            <li key={`${r.label}-${i}`}>
              <span className="rb-label">{r.label}</span>
              <span className="rb-track">
                <span
                  className="rb-fill"
                  data-tone={tone === "rival" ? "rival" : undefined}
                  style={{ width: `${Math.max(0.8, (100 * p) / max)}%`, transition: "none" }}
                />
              </span>
              <span className="rb-val">{(100 * p).toFixed(digits ?? (p < 0.1 ? 1 : 0))}%</span>
            </li>
          );
        })}
      </ul>
      {rows.length > limit && <p className="figure-note">{rows.length - limit} smaller levels not shown.</p>}
    </>
  );
}
