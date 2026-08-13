"use client";

/* Offer a block of capacity into a uniform-price market.
 *
 * POST /v1/solve/electricity — body mirrors sq.ElectricityProblem(costs=,
 * offers=, capacities=, demand=, precision=).
 */

import { useMemo, useState } from "react";
import { gridLevels, sweepPoints, useSolve, useSweep, type ElectricitySolution } from "../../../lib/problems";
import { OfferStack } from "../charts/OfferStack";
import { Answer, Bars, Controls, Curve, Field, Figure, ModelLine, Sweep, money, money0, num, pct, sig } from "./ui";

interface Inputs {
  yourCost: number;
  rivalCost: number;
  low: number;
  high: number;
  step: number;
  yourCapacity: number;
  rivalCapacity: number;
  demand: number;
  precision: number;
}

const DEFAULTS: Inputs = {
  yourCost: 20,
  rivalCost: 22,
  low: 20,
  high: 60,
  step: 5,
  yourCapacity: 100,
  rivalCapacity: 100,
  demand: 80,
  precision: 0.05,
};

function bodyFor(v: Inputs) {
  return {
    costs: [v.yourCost, v.rivalCost],
    offers_range: [v.low, v.high, v.step],
    capacities: [v.yourCapacity, v.rivalCapacity],
    demand: v.demand,
    precision: v.precision,
  };
}

const SWEEPS = [
  { key: "demand", label: "Demand (MW)", min: 20, max: 190, step: 10 },
  { key: "rivalCost", label: "Rival marginal cost", min: 10, max: 50, step: 2 },
  { key: "yourCapacity", label: "Your capacity (MW)", min: 40, max: 200, step: 10 },
  { key: "precision", label: "Precision", min: 0.005, max: 0.3, step: 0.005 },
] as const;

const mwh = (v: number) => `$${num(v, 2)}/MWh`;

export function ElectricitySolver() {
  const [v, setV] = useState<Inputs>(DEFAULTS);
  const [sweepKey, setSweepKey] = useState<string>("demand");
  const set = (patch: Partial<Inputs>) => setV((old) => ({ ...old, ...patch }));

  const levels = gridLevels(v.low, v.high, v.step);
  const valid = levels >= 2 && levels <= 60;

  const body = useMemo(() => bodyFor(v), [v]);
  const { data, error, busy } = useSolve<ElectricitySolution>("electricity", body);

  const spec = SWEEPS.find((s) => s.key === sweepKey) ?? SWEEPS[0];
  const points = useMemo(
    () => sweepPoints(v[spec.key as keyof Inputs] as number, spec.min, spec.max, spec.step),
    [v, spec],
  );
  const sweepBodies = useMemo(
    () => (valid ? points.map((p) => bodyFor({ ...v, [spec.key]: p })) : []),
    [points, v, spec, valid],
  );
  const swept = useSweep<ElectricitySolution>("electricity", sweepBodies);

  const rows =
    swept.length === points.length
      ? points.map((p, i) => {
          const s = swept[i];
          return {
            x: spec.key === "precision" ? sig(p) : num(p, 0),
            y: s ? mwh(s.offer) : "—",
            value: s ? mwh(s.clearing_price) : "—",
            changed: !!s && !!data && Math.abs(s.offer - data.offer) > 1e-9,
          };
        })
      : [];

  const markIndex = data ? data.offers.findIndex((o) => Math.abs(o - data.offer) < 1e-9) : -1;
  const tight = v.demand > Math.max(v.yourCapacity, v.rivalCapacity);

  return (
    <div className="studio">
      <Answer
        headline={data ? `Offer at ${mwh(data.offer)}.` : "Solving…"}
        busy={busy}
        error={error ?? (valid ? null : "The offer grid must give between 2 and 60 levels.")}
      >
        {data && (
          <>
            <Figure label="Expected clearing price" value={mwh(data.clearing_price)} tween={data.clearing_price} format={mwh} note="uniform price paid to all dispatched capacity" />
            <Figure label="Expected revenue" value={money0(data.revenue)} tween={data.revenue} format={money0} note="clearing price times your dispatch" tone="neutral" />
            <Figure label="Dispatch probability" value={pct(data.dispatch_probability)} tween={data.dispatch_probability} format={(x) => pct(x)} note="share of your capacity called" tone="neutral" />
          </>
        )}
      </Answer>

      {data && (
        <ModelLine>
          Uniform-price auction, 2 generators, {data.offers.length} offer levels, demand{" "}
          {num(data.demand, 0)} MW against {num(data.capacities[0] + data.capacities[1], 0)} MW of capacity,
          precision {sig(data.precision)}. Expected profit {money0(data.profit)}.
          {data.success ? "" : ` ${data.message}`}
        </ModelLine>
      )}

      {data && (
        <section className="card">
          <h3>The stack, and where it crosses demand</h3>
          <OfferStack
            units={[
              {
                label: "you",
                offer: data.offer,
                cost: data.costs[data.generator],
                capacity: data.capacities[data.generator],
                mine: true,
              },
              {
                label: "other unit",
                offer: data.costs[1 - data.generator],
                cost: data.costs[1 - data.generator],
                capacity: data.capacities[1 - data.generator],
                mine: false,
              },
            ]}
            demand={data.demand}
            clearingPrice={data.clearing_price}
            formatPrice={(x) => money(x, 0)}
            formatMW={(x) => `${num(x, 0)} MW`}
          />
          <p className="chart-note">
            Only your offer is a decision here, so only your block carries a mark-up over cost; the other unit is drawn
            at its own marginal cost. The horizontal rule is the expected clearing price returned by the solve.
          </p>
        </section>
      )}

      <div className="answer-cols">
        <section className="card">
          <h3>Profit at every offer on the stack</h3>
          {data && (
            <Curve
              x={data.offers}
              y={data.profit_curve}
              markIndex={markIndex}
              formatX={(o) => money(o, 0)}
              formatY={money0}
              xLabel="your offer price"
              yLabel="expected profit"
            />
          )}
          {data && (
            <table className="alt-table">
              <thead>
                <tr>
                  <th scope="col">Offer</th>
                  <th scope="col">Profit</th>
                  <th scope="col">Gives up</th>
                </tr>
              </thead>
              <tbody>
                {data.offers
                  .map((o, i) => ({ o, profit: data.profit_curve[i] }))
                  .sort((a, b) => b.profit - a.profit)
                  .slice(0, 4)
                  .map((r) => (
                    <tr key={r.o} data-us={Math.abs(r.o - data.offer) < 1e-9}>
                      <th scope="row">{mwh(r.o)}</th>
                      <td>{money0(r.profit)}</td>
                      <td className="alt-cost">
                        {Math.abs(r.profit - data.profit) < 1e-9 ? "—" : `−${money0(data.profit - r.profit)}`}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          )}
        </section>

        <section className="card">
          <h3>Where the market clears</h3>
          {data && (
            <Bars rows={data.clearing_price_distribution.map(([price, p]) => ({ label: mwh(price), p }))} />
          )}
          {data && (
            <>
              <h3 style={{ marginTop: "1.2rem" }}>Your offer stack</h3>
              <Bars rows={data.offer_curve.map(([price, p]) => ({ label: mwh(price), p }))} limit={6} />
            </>
          )}
        </section>
      </div>

      <Sweep
        inputLabel={spec.label}
        choices={SWEEPS.map((s) => ({ key: s.key, label: s.label }))}
        chosen={sweepKey}
        onChoose={setSweepKey}
        rows={rows}
        outputLabel="Offer"
        valueLabel="Clearing price"
      />

      <Controls>
        <Field label="Your marginal cost" value={v.yourCost} onChange={(x) => set({ yourCost: x })} min={0} max={120} step={0.5} format={mwh} />
        <Field label="Rival marginal cost" value={v.rivalCost} onChange={(x) => set({ rivalCost: x })} min={0} max={120} step={0.5} format={mwh} />
        <Field
          label="Demand"
          value={v.demand}
          onChange={(x) => set({ demand: x })}
          min={5}
          max={400}
          step={5}
          format={(x) => `${num(x, 0)} MW`}
          help={
            tight
              ? "Above one generator's capacity, so neither can serve the load alone and both are dispatched."
              : "Below one generator's capacity, so either could serve the load alone."
          }
        />
        <Field label="Your capacity" value={v.yourCapacity} onChange={(x) => set({ yourCapacity: x })} min={10} max={400} step={5} format={(x) => `${num(x, 0)} MW`} />
        <Field label="Rival capacity" value={v.rivalCapacity} onChange={(x) => set({ rivalCapacity: x })} min={10} max={400} step={5} format={(x) => `${num(x, 0)} MW`} />
        <Field
          label="Precision"
          value={v.precision}
          onChange={(x) => set({ precision: x })}
          min={0.002}
          max={0.5}
          step={0.002}
          format={sig}
          help="Per dollar of profit. Higher means both generators offer closer to their own optimum every interval."
        />
        <Field label="Lowest offer on the stack" value={v.low} onChange={(x) => set({ low: x })} min={0} max={200} step={1} format={(x) => money(x, 0)} />
        <Field label="Highest offer on the stack" value={v.high} onChange={(x) => set({ high: x })} min={5} max={500} step={1} format={(x) => money(x, 0)} />
        <Field
          label="Offer step"
          value={v.step}
          onChange={(x) => set({ step: x })}
          min={0.5}
          max={50}
          step={0.5}
          format={(x) => money(x, 2)}
          help={`${levels} levels on the stack.`}
        />
      </Controls>
    </div>
  );
}
