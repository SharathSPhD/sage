"use client";

/* Sealed-bid tender or sale.
 *
 * POST /v1/solve/auction — costs= for a procurement tender (lowest eligible
 * bid wins), values= for a sale (highest wins). Body mirrors
 * sq.AuctionProblem.
 */

import { useMemo, useState } from "react";
import { gridLevels, sweepPoints, useSolve, useSweep, type AuctionSolution } from "../../../lib/problems";
import { Answer, Bars, Controls, Curve, Field, Figure, ModelLine, Sweep, money0, pct, sig } from "./ui";

type Mode = "procurement" | "sale";

interface Inputs {
  mode: Mode;
  yours: number;
  rival: number;
  low: number;
  high: number;
  step: number;
  reserve: number;
  precision: number;
}

const TENDER: Inputs = {
  mode: "procurement",
  yours: 85000,
  rival: 88000,
  low: 88000,
  high: 116000,
  step: 4000,
  reserve: 112000,
  precision: 0.0005,
};

const SALE: Inputs = {
  mode: "sale",
  yours: 120000,
  rival: 115000,
  low: 60000,
  high: 120000,
  step: 5000,
  reserve: 70000,
  precision: 0.0005,
};

function bodyFor(v: Inputs) {
  const listed = [v.yours, v.rival];
  return {
    ...(v.mode === "procurement" ? { costs: listed } : { values: listed }),
    grid_range: [v.low, v.high, v.step],
    reserve: v.reserve,
    precision: v.precision,
  };
}

const SWEEPS = [
  { key: "rival", label: "Rival's cost or value", min: 60000, max: 120000, step: 2000 },
  { key: "yours", label: "Your cost or value", min: 60000, max: 120000, step: 2000 },
  { key: "reserve", label: "Reserve", min: 70000, max: 130000, step: 2000 },
  { key: "precision", label: "Precision", min: 0.00005, max: 0.003, step: 0.00005 },
] as const;

export function AuctionSolver({ mode: fixedMode }: { mode?: Mode }) {
  const [v, setV] = useState<Inputs>(fixedMode === "sale" ? SALE : TENDER);
  const [sweepKey, setSweepKey] = useState<string>("rival");
  const set = (patch: Partial<Inputs>) => setV((old) => ({ ...old, ...patch }));

  const levels = gridLevels(v.low, v.high, v.step);
  const valid = levels >= 2 && levels <= 60;

  const body = useMemo(() => bodyFor(v), [v]);
  const { data, error, busy } = useSolve<AuctionSolution>("auction", body);

  const spec = SWEEPS.find((s) => s.key === sweepKey) ?? SWEEPS[0];
  const points = useMemo(
    () => sweepPoints(v[spec.key as keyof Inputs] as number, spec.min, spec.max, spec.step),
    [v, spec],
  );
  const sweepBodies = useMemo(
    () => (valid ? points.map((p) => bodyFor({ ...v, [spec.key]: p })) : []),
    [points, v, spec, valid],
  );
  const swept = useSweep<AuctionSolution>("auction", sweepBodies);

  const rows =
    swept.length === points.length
      ? points.map((p, i) => {
          const s = swept[i];
          return {
            x: spec.key === "precision" ? sig(p) : money0(p),
            y: s ? money0(s.bid) : "—",
            value: s ? money0(s.surplus) : "—",
            changed: !!s && !!data && Math.abs(s.bid - data.bid) > 1e-9,
          };
        })
      : [];

  const markIndex = data ? data.bid_grid.findIndex((b) => Math.abs(b - data.bid) < 1e-9) : -1;
  const winning = v.mode === "procurement" ? "lowest eligible bid wins" : "highest eligible bid wins";

  return (
    <div className="studio">
      {!fixedMode && (
        <div className="situation-picker" role="group" aria-label="Auction direction">
          {(["procurement", "sale"] as Mode[]).map((m) => (
            <button
              key={m}
              type="button"
              data-on={v.mode === m}
              aria-pressed={v.mode === m}
              onClick={() => setV(m === "sale" ? SALE : TENDER)}
            >
              {m === "procurement" ? "Tender (you supply)" : "Sale (you buy)"}
            </button>
          ))}
        </div>
      )}

      <Answer
        headline={data ? `Bid ${money0(data.bid)}.` : "Solving…"}
        busy={busy}
        error={error ?? (valid ? null : "The bid grid must give between 2 and 60 levels.")}
      >
        {data && (
          <>
            <Figure label="Expected surplus" value={money0(data.surplus)} note="win probability times margin" />
            <Figure label="Win probability" value={pct(data.win_probability)} note="at this bid" tone="neutral" />
            <Figure
              label="Expected clearing bid"
              value={money0(data.expected_clearing_bid)}
              note="mean of the winning bid"
              tone="neutral"
            />
          </>
        )}
      </Answer>

      {data && (
        <ModelLine>
          Sealed bid, {data.n_bidders} bidders, {data.bid_grid.length} levels, {winning}, reserve{" "}
          {data.reserve === null ? "none" : money0(data.reserve)}, precision {sig(data.precision)}.
          {data.success ? "" : ` ${data.message}`}
        </ModelLine>
      )}

      <div className="answer-cols">
        <section className="card">
          <h3>Surplus at every bid on the grid</h3>
          {data && (
            <Curve
              x={data.bid_grid}
              y={data.surplus_curve}
              markIndex={markIndex}
              formatX={money0}
              formatY={money0}
              xLabel="your bid"
              yLabel="expected surplus"
            />
          )}
          {data && (
            <table className="alt-table">
              <thead>
                <tr>
                  <th scope="col">Bid</th>
                  <th scope="col">Surplus</th>
                  <th scope="col">Win</th>
                </tr>
              </thead>
              <tbody>
                {data.bid_grid
                  .map((b, i) => ({ b, s: data.surplus_curve[i], w: data.win_curve[i] }))
                  .sort((a, b) => b.s - a.s)
                  .slice(0, 4)
                  .map((r) => (
                    <tr key={r.b} data-us={Math.abs(r.b - data.bid) < 1e-9}>
                      <th scope="row">{money0(r.b)}</th>
                      <td>{money0(r.s)}</td>
                      <td>{pct(r.w)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          )}
        </section>

        <section className="card">
          <h3>Rival&apos;s bid</h3>
          {data && (
            <Bars rows={data.rival_bids[0].map((p, i) => ({ label: money0(data.bid_grid[i]), p }))} />
          )}
        </section>
      </div>

      <Sweep
        inputLabel={spec.label}
        choices={SWEEPS.map((s) => ({ key: s.key, label: s.label }))}
        chosen={sweepKey}
        onChoose={setSweepKey}
        rows={rows}
        outputLabel="Bid"
        valueLabel="Surplus"
      />

      <Controls>
        <Field
          label={v.mode === "procurement" ? "Your cost to deliver" : "What the contract is worth to you"}
          value={v.yours}
          onChange={(x) => set({ yours: x })}
          min={40000}
          max={150000}
          step={1000}
          format={money0}
        />
        <Field
          label={v.mode === "procurement" ? "What you think it costs them" : "What you think it is worth to them"}
          value={v.rival}
          onChange={(x) => set({ rival: x })}
          min={40000}
          max={150000}
          step={1000}
          format={money0}
        />
        <Field
          label={v.mode === "procurement" ? "Buyer's walk-away price" : "Seller's reserve"}
          value={v.reserve}
          onChange={(x) => set({ reserve: x })}
          min={40000}
          max={160000}
          step={1000}
          format={money0}
          help={v.mode === "procurement" ? "Bids above this are not eligible." : "Bids below this are not eligible."}
        />
        <Field
          label="Precision"
          value={v.precision}
          onChange={(x) => set({ precision: x })}
          min={0.00005}
          max={0.005}
          step={0.00005}
          format={sig}
          help="Per dollar of surplus. Higher means the rival bids closer to their own optimum every time."
        />
        <Field label="Lowest bid on the grid" value={v.low} onChange={(x) => set({ low: x })} min={20000} max={200000} step={1000} format={money0} />
        <Field label="Highest bid on the grid" value={v.high} onChange={(x) => set({ high: x })} min={30000} max={250000} step={1000} format={money0} />
        <Field
          label="Bid step"
          value={v.step}
          onChange={(x) => set({ step: x })}
          min={500}
          max={20000}
          step={500}
          format={money0}
          help={`${levels} levels on the grid.`}
        />
      </Controls>
    </div>
  );
}
