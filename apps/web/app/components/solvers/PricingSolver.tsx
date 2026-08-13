"use client";

/* Set a price against a rival who is setting one too.
 *
 * POST /v1/solve/pricing — the body mirrors sq.PricingProblem(costs=, grid=,
 * demand=, precision=). Everything on screen is a field of the returned
 * Solution; nothing is recomputed here.
 */

import { useMemo, useState } from "react";
import {
  gridLevels,
  sweepPoints,
  useSolve,
  useSweep,
  type PricingSolution,
} from "../../../lib/problems";
import { PriceLadder } from "../charts/PriceLadder";
import { Answer, Bars, Controls, Curve, Field, Figure, ModelLine, Sweep, money, money0, num, sig } from "./ui";

interface Inputs {
  yourCost: number;
  rivalCost: number;
  low: number;
  high: number;
  step: number;
  sensitivity: number;
  advantage: number;
  outside: number;
  units: number;
  precision: number;
}

const DEFAULTS: Inputs = {
  yourCost: 1.0,
  rivalCost: 1.05,
  low: 1.09,
  high: 1.89,
  step: 0.1,
  sensitivity: 3.6,
  advantage: 0,
  outside: 1.65,
  units: 400,
  precision: 1.5,
};

/*
 * The logit quality terms are set so that the outside option — private label,
 * a different format, walking out — sits at the price the user names. Utility
 * is then `sensitivity * (outside - price)` plus your brand advantage, and a
 * product priced at the outside option holds the same share as it.
 */
function bodyFor(v: Inputs) {
  const anchor = v.sensitivity * v.outside;
  return {
    costs: [v.yourCost, v.rivalCost],
    grid_range: [v.low, v.high, v.step],
    demand: {
      kind: "logit",
      price_sensitivity: v.sensitivity,
      quality: [anchor + v.advantage, anchor],
      market_size: v.units,
    },
    precision: v.precision,
  };
}

const SWEEPS = [
  { key: "yourCost", label: "Your unit cost", min: 0.6, max: 1.4, step: 0.05 },
  { key: "rivalCost", label: "Rival unit cost", min: 0.6, max: 1.4, step: 0.05 },
  { key: "sensitivity", label: "Price sensitivity", min: 1, max: 8, step: 0.5 },
  { key: "outside", label: "Outside option price", min: 1.2, max: 2.6, step: 0.1 },
  { key: "advantage", label: "Brand advantage", min: -0.6, max: 0.6, step: 0.1 },
  { key: "precision", label: "Precision", min: 0.2, max: 12, step: 0.2 },
] as const;

export function PricingSolver() {
  const [v, setV] = useState<Inputs>(DEFAULTS);
  const [sweepKey, setSweepKey] = useState<string>("rivalCost");
  const set = (patch: Partial<Inputs>) => setV((old) => ({ ...old, ...patch }));

  const levels = gridLevels(v.low, v.high, v.step);
  const valid = levels >= 2 && levels <= 60 && v.high >= v.low && v.step > 0;

  const body = useMemo(() => bodyFor(v), [v]);
  const { data, error, busy } = useSolve<PricingSolution>("pricing", body);

  const spec = SWEEPS.find((s) => s.key === sweepKey) ?? SWEEPS[0];
  const points = useMemo(
    () => sweepPoints(v[spec.key as keyof Inputs] as number, spec.min, spec.max, spec.step),
    [v, spec],
  );
  const sweepBodies = useMemo(
    () => (valid ? points.map((p) => bodyFor({ ...v, [spec.key]: p })) : []),
    [points, v, spec, valid],
  );
  const swept = useSweep<PricingSolution>("pricing", sweepBodies);

  const rows =
    swept.length === points.length
      ? points.map((p, i) => {
          const s = swept[i];
          return {
            x: spec.key === "precision" || spec.key === "sensitivity" ? num(p, 1) : money(p),
            y: s ? money(s.price) : "—",
            value: s ? money0(s.profit) : "—",
            changed: !!s && !!data && Math.abs(s.price - data.price) > 1e-9,
          };
        })
      : [];

  const markIndex = data ? data.price_grid.findIndex((p) => Math.abs(p - data.price) < 1e-9) : -1;

  return (
    <div className="studio">
      <Answer
        headline={data ? `Price at ${money(data.price)}.` : "Solving…"}
        busy={busy}
        error={error ?? (valid ? null : "The price grid must give between 2 and 60 levels.")}
      >
        {data && (
          <>
            <Figure label="Expected profit" value={money0(data.profit)} tween={data.profit} format={money0} note="per store-week at this price" />
            <Figure label="Margin" value={money(data.margin)} tween={data.margin} format={money} note="price minus your unit cost" tone="neutral" />
            <Figure
              label="Rival's expected price"
              value={money(data.expected_rival_prices[0])}
              tween={data.expected_rival_prices[0]}
              format={money}
              note="mean of their price distribution"
              tone="neutral"
            />
          </>
        )}
      </Answer>

      {data && (
        <ModelLine>
          Logit demand with an outside option at {money(v.outside)}, {data.n_firms} firms,{" "}
          {data.price_grid.length} price levels, {v.units.toLocaleString()} units, precision{" "}
          {sig(data.precision)}. Own-price elasticity{" "}
          {num(data.elasticities[0][0], 2)}, cross-price {sig(data.elasticities[0][1])}.
          {data.success ? "" : ` ${data.message}`}
        </ModelLine>
      )}

      <div className="answer-cols">
        <section className="card">
          <h3>Profit at every price on the grid</h3>
          {data && (
            <Curve
              x={data.price_grid}
              y={data.profit_curve}
              markIndex={markIndex}
              formatX={(p) => money(p)}
              formatY={(p) => money0(p)}
              xLabel="your price"
              yLabel="expected profit"
            />
          )}
        </section>

        <section className="card">
          <h3>The price ladder</h3>
          {data && (
            <PriceLadder
              levels={data.price_grid}
              profit={data.profit_curve}
              rival={data.rival_prices[0]}
              chosen={markIndex}
              formatLevel={(p) => money(p)}
              formatProfit={money0}
            />
          )}
        </section>
      </div>

      <div className="answer-cols">
        <section className="card">
          <h3>Rival&apos;s price</h3>
          {data && (
            <Bars
              rows={data.rival_prices[0].map((p, i) => ({ label: money(data.price_grid[i]), p }))}
              tone="rival"
            />
          )}
          {data && (
            <table className="alt-table">
              <caption className="visually-hidden">Elasticities</caption>
              <tbody>
                <tr>
                  <th scope="row">Own-price elasticity</th>
                  <td>{num(data.elasticities[0][0], 2)}</td>
                </tr>
                <tr>
                  <th scope="row">Cross-price elasticity</th>
                  <td>{sig(data.elasticities[0][1])}</td>
                </tr>
                <tr>
                  <th scope="row">Your price, as a distribution</th>
                  <td>{money(data.price_grid[data.own_price_distribution.indexOf(Math.max(...data.own_price_distribution))])} most likely</td>
                </tr>
              </tbody>
            </table>
          )}
        </section>
      </div>

      <Sweep
        inputLabel={spec.label}
        choices={SWEEPS.map((s) => ({ key: s.key, label: s.label }))}
        chosen={sweepKey}
        onChoose={setSweepKey}
        rows={rows}
        outputLabel="Price"
        valueLabel="Profit"
      />

      <Controls>
        <Field
          label="Your unit cost"
          value={v.yourCost}
          onChange={(x) => set({ yourCost: x })}
          min={0.4}
          max={1.6}
          step={0.01}
          format={money}
        />
        <Field
          label="Rival unit cost"
          value={v.rivalCost}
          onChange={(x) => set({ rivalCost: x })}
          min={0.4}
          max={1.6}
          step={0.01}
          format={money}
        />
        <Field
          label="Price sensitivity"
          value={v.sensitivity}
          onChange={(x) => set({ sensitivity: x })}
          min={0.5}
          max={9}
          step={0.1}
          format={(x) => num(x, 1)}
          help="Logit price coefficient. Higher means buyers switch harder for the same 10c."
        />
        <Field
          label="Brand advantage over the rival"
          value={v.advantage}
          onChange={(x) => set({ advantage: x })}
          min={-1}
          max={1}
          step={0.05}
          format={(x) => num(x, 2)}
          help="Difference in logit quality. Zero means the two products are interchangeable at equal prices."
        />
        <Field
          label="Price of the next best thing on the shelf"
          value={v.outside}
          onChange={(x) => set({ outside: x })}
          min={0.8}
          max={4}
          step={0.05}
          format={money}
          help="Private label, a different pack size, or leaving with neither. Sets where the outside option sits."
        />
        <Field
          label="Category units per store-week"
          value={v.units}
          onChange={(x) => set({ units: x })}
          min={50}
          max={5000}
          step={10}
          format={(x) => Math.round(x).toLocaleString()}
        />
        <Field
          label="Precision"
          value={v.precision}
          onChange={(x) => set({ precision: x })}
          min={0.05}
          max={20}
          step={0.05}
          format={(x) => num(x, 2)}
          help="How exactly each firm optimises: near zero is close to random, high is close to exact best response. Scales with your profit units."
        />
        <Field label="Lowest price on the grid" value={v.low} onChange={(x) => set({ low: x })} min={0.5} max={3} step={0.01} format={money} />
        <Field label="Highest price on the grid" value={v.high} onChange={(x) => set({ high: x })} min={0.6} max={5} step={0.01} format={money} />
        <Field
          label="Price step"
          value={v.step}
          onChange={(x) => set({ step: x })}
          min={0.01}
          max={0.5}
          step={0.01}
          format={money}
          help={`${levels} levels on the grid.`}
        />
      </Controls>
    </div>
  );
}
