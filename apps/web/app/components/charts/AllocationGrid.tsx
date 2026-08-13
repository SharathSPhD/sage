"use client";

/* Colonel Blotto as a grid.
 *
 * Every feasible split of the budget is a cell: how much goes to the first
 * account across, how much to the second down, the rest to the third. Cell
 * shade is the weight the solver puts on that split — one hue, light to
 * dark, so it reads as a magnitude and not as six unrelated categories.
 * Weights tween, so moving a budget makes the mass migrate across the grid.
 */

import { Fragment, useMemo } from "react";
import { useTweenArray } from "../../../lib/anim";
import { TableTwin } from "./primitives";
import { Shade, ShadeLegend } from "./Shade";

export function AllocationGrid({
  allocations,
  weights,
  budget,
  title,
  fieldNames = ["A", "B", "C"],
}: {
  allocations: number[][];
  weights: number[];
  budget: number;
  title: string;
  fieldNames?: string[];
}) {
  const smooth = useTweenArray(weights, 420);
  const max = Math.max(...smooth, 1e-9);

  const cells = useMemo(() => {
    const map = new Map<string, { i: number; alloc: number[] }>();
    allocations.forEach((alloc, i) => map.set(`${alloc[0]}|${alloc[1] ?? 0}`, { i, alloc }));
    return map;
  }, [allocations]);

  const n = budget + 1;
  const threeFields = (allocations[0]?.length ?? 3) >= 3;

  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between", marginBottom: "0.6rem" }}>
        <h3 style={{ margin: 0 }}>{title}</h3>
        <ShadeLegend />
      </div>
      <div
        className="blotto-grid"
        style={{ gridTemplateColumns: `1.6rem repeat(${n}, minmax(0, 1fr))` }}
        role="img"
        aria-label={`${title}. Each cell is a split of ${budget} units; darker means more weight. The table below carries the numbers.`}
      >
        <span />
        {Array.from({ length: n }, (_, b) => (
          <span key={`h${b}`} className="blotto-legend" style={{ justifyContent: "center", fontFamily: "var(--mono)" }}>
            {b}
          </span>
        ))}
        {Array.from({ length: n }, (_, a) => (
          <Fragment key={`row-${a}`}>
            <span
              className="blotto-legend"
              style={{ justifyContent: "flex-end", paddingRight: "0.3rem", fontFamily: "var(--mono)" }}
            >
              {a}
            </span>
            {Array.from({ length: n }, (_, b) => {
              const hit = cells.get(`${a}|${b}`);
              const feasible = !!hit && (!threeFields || a + b <= budget);
              const p = hit ? smooth[hit.i] ?? 0 : 0;
              return (
                <span
                  key={`c${a}-${b}`}
                  className="blotto-cell"
                  style={{
                    background: feasible ? "var(--surface-2)" : "transparent",
                    border: feasible ? "none" : "1px dashed var(--border)",
                    color: "var(--text)",
                  }}
                  title={
                    feasible && hit
                      ? `${hit.alloc.join(" · ")} — ${(100 * p).toFixed(1)}%`
                      : "not a feasible split"
                  }
                >
                  {feasible && <Shade t={p / max} />}
                  <span style={{ position: "relative" }}>
                    {feasible && p / max > 0.25 ? `${Math.round(100 * p)}` : ""}
                  </span>
                </span>
              );
            })}
          </Fragment>
        ))}
      </div>
      <p className="chart-note">
        Across: units to {fieldNames[0]}. Down: units to {fieldNames[1]}. The rest goes to {fieldNames[2] ?? "the last account"}.
        Cells with a dashed edge are not feasible splits; the deeper the shade, the more weight the split carries, and
        the numbers on the heavier cells are percentages.
      </p>
      <TableTwin
        caption="Every split and its weight"
        columns={["Split", "Weight"]}
        rows={allocations
          .map((a, i) => ({ a, p: weights[i] ?? 0 }))
          .sort((x, y) => y.p - x.p)
          .map((r) => [r.a.join(" · "), `${(100 * r.p).toFixed(2)}%`])}
      />
    </div>
  );
}
