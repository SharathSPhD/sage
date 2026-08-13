"use client";

/* A coordination payoff table with its basins.
 *
 * Each cell carries the two payoffs and is shaded by how much of the joint
 * solution lands there — the basin. One hue, light to dark, tweened, so
 * turning up precision visibly pulls the mass into a corner and turning it
 * down floods the table. The row and column margins carry each side's own
 * distribution, so the reader can see who is committing and who is hedging.
 */

import { useTweenArray } from "../../../lib/anim";
import { TableTwin } from "./primitives";
import { Shade, ShadeLegend } from "./Shade";

export function PayoffMatrix({
  rowLabels,
  colLabels,
  payoffs,
  rowMix,
  colMix,
  rowName = "You",
  colName = "Them",
  format = (v: number) => v.toFixed(1),
}: {
  rowLabels: string[];
  colLabels: string[];
  /** payoffs[i][j] = [your payoff, their payoff] */
  payoffs: [number, number][][];
  rowMix: number[];
  colMix: number[];
  rowName?: string;
  colName?: string;
  format?: (v: number) => string;
}) {
  const joint: number[] = [];
  rowMix.forEach((r) => colMix.forEach((c) => joint.push(r * c)));
  const smooth = useTweenArray(joint, 420);
  const rows = useTweenArray(rowMix, 420);
  const cols = useTweenArray(colMix, 420);
  const max = Math.max(...smooth, 1e-9);

  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between", marginBottom: "0.5rem" }}>
        <span className="panel-label" style={{ margin: 0 }}>
          {colName} choose →
        </span>
        <ShadeLegend low="rarely" high="often" />
      </div>
      <table className="payoff-matrix">
        <thead>
          <tr>
            <th scope="col">
              <span className="visually-hidden">{rowName} choose</span>
            </th>
            {colLabels.map((c, j) => (
              <th key={c} scope="col">
                {c}
                <span style={{ display: "block", fontFamily: "var(--mono)", color: "var(--text-3)", fontWeight: 400 }}>
                  {(100 * (cols[j] ?? 0)).toFixed(0)}%
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rowLabels.map((r, i) => (
            <tr key={r}>
              <th scope="row" style={{ textAlign: "right" }}>
                {r}
                <span style={{ display: "block", fontFamily: "var(--mono)", color: "var(--text-3)", fontWeight: 400 }}>
                  {(100 * (rows[i] ?? 0)).toFixed(0)}%
                </span>
              </th>
              {colLabels.map((c, j) => {
                const p = smooth[i * colLabels.length + j] ?? 0;
                const pair = payoffs[i]?.[j] ?? [0, 0];
                return (
                  <td key={c} style={{ position: "relative", color: "var(--text)" }}>
                    <Shade t={p / max} />
                    <span style={{ position: "relative" }}>
                      {format(pair[0])} , {format(pair[1])}
                      <span className="cell-weight">{(100 * p).toFixed(0)}%</span>
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="chart-note">
        Each cell reads <em>your payoff, their payoff</em>. The shade is how often the pair actually lands there; the
        percentage under a row or column heading is that side&apos;s own distribution.
      </p>
      <TableTwin
        caption="Payoffs and joint weight, as numbers"
        columns={[`${rowName} play`, `${colName} play`, "Your payoff", "Their payoff", "Weight"]}
        rows={rowLabels.flatMap((r, i) =>
          colLabels.map((c, j) => [
            r,
            c,
            format(payoffs[i]?.[j]?.[0] ?? 0),
            format(payoffs[i]?.[j]?.[1] ?? 0),
            `${(100 * (joint[i * colLabels.length + j] ?? 0)).toFixed(1)}%`,
          ]),
        )}
      />
    </div>
  );
}
