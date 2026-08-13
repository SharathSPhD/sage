"use client";

/* Split a fixed budget across contested fields against a rival doing the same.
 *
 * POST /v1/solve/allocation — body mirrors sq.AllocationProblem(budget=,
 * field_values=, rival_budget=, precision=). Every split is enumerated, so the
 * distributions below are exact over the whole action set.
 */

import { useMemo, useState } from "react";
import { sweepPoints, useSolve, useSweep, type AllocationSolution } from "../../../lib/problems";
import { Answer, Bars, Controls, Field, Figure, ModelLine, Sweep, num, pct } from "./ui";

interface Inputs {
  budget: number;
  rivalBudget: number;
  a: number;
  b: number;
  c: number;
  precision: number;
}

const DEFAULTS: Inputs = { budget: 5, rivalBudget: 5, a: 1, b: 1, c: 2, precision: 2 };

function bodyFor(v: Inputs) {
  return {
    budget: Math.round(v.budget),
    rival_budget: Math.round(v.rivalBudget),
    field_values: [v.a, v.b, v.c],
    precision: v.precision,
  };
}

const SWEEPS = [
  { key: "rivalBudget", label: "Rival budget", min: 1, max: 10, step: 1 },
  { key: "budget", label: "Your budget", min: 1, max: 10, step: 1 },
  { key: "c", label: "Value of account C", min: 0.5, max: 5, step: 0.5 },
  { key: "precision", label: "Precision", min: 0.2, max: 8, step: 0.2 },
] as const;

const split = (a: number[]) => a.join(" · ");

export function AllocationSolver() {
  const [v, setV] = useState<Inputs>(DEFAULTS);
  const [sweepKey, setSweepKey] = useState<string>("rivalBudget");
  const set = (patch: Partial<Inputs>) => setV((old) => ({ ...old, ...patch }));

  const body = useMemo(() => bodyFor(v), [v]);
  const { data, error, busy } = useSolve<AllocationSolution>("allocation", body);

  const spec = SWEEPS.find((s) => s.key === sweepKey) ?? SWEEPS[0];
  const points = useMemo(
    () => sweepPoints(v[spec.key as keyof Inputs] as number, spec.min, spec.max, spec.step),
    [v, spec],
  );
  const sweepBodies = useMemo(() => points.map((p) => bodyFor({ ...v, [spec.key]: p })), [points, v, spec]);
  const swept = useSweep<AllocationSolution>("allocation", sweepBodies);

  const rows =
    swept.length === points.length
      ? points.map((p, i) => {
          const s = swept[i];
          return {
            x: spec.key === "precision" || spec.key === "c" ? num(p, 1) : String(Math.round(p)),
            y: s ? split(s.allocation) : "—",
            value: s ? pct(s.win_probability) : "—",
            changed: !!s && !!data && split(s.allocation) !== split(data.allocation),
          };
        })
      : [];

  const spread = data ? 1 - Math.max(...data.allocation_distribution) : 0;

  return (
    <div className="studio">
      <Answer
        headline={data ? `Put ${split(data.allocation)} across the three accounts.` : "Solving…"}
        busy={busy}
        error={error}
      >
        {data && (
          <>
            <Figure label="Win probability" value={pct(data.win_probability)} note="of taking more than half the total value" />
            <Figure label="Expected value" value={num(data.expected_value, 2)} note="value captured on average" tone="neutral" />
            <Figure
              label="Splits worth using"
              value={String(data.allocation_distribution.filter((p) => p > 0.02).length)}
              note={`of ${data.allocations.length} possible splits carry more than 2% weight`}
              tone="neutral"
            />
          </>
        )}
      </Answer>

      {data && (
        <ModelLine>
          Blotto, {data.n_fields} fields worth {data.field_values.map((x) => num(x, 1)).join(", ")}, your budget{" "}
          {data.budget} against {data.rival_budget}, {data.allocations.length} splits enumerated, precision{" "}
          {num(data.precision, 2)}.{data.success ? "" : ` ${data.message}`}
        </ModelLine>
      )}

      {data && spread > 0.6 && (
        <p className="figure-note">
          No single split carries most of the weight: the distribution below is the answer, not the top row.
        </p>
      )}

      <div className="answer-cols">
        <section className="card">
          <h3>Your splits, by weight</h3>
          {data && (
            <Bars rows={data.allocation_distribution.map((p, i) => ({ label: split(data.allocations[i]), p }))} limit={8} />
          )}
        </section>
        <section className="card">
          <h3>Rival&apos;s splits, by weight</h3>
          {data && (
            <Bars rows={data.rival_distribution.map((p, i) => ({ label: split(data.rival_allocations[i]), p }))} limit={8} />
          )}
        </section>
      </div>

      <Sweep
        inputLabel={spec.label}
        choices={SWEEPS.map((s) => ({ key: s.key, label: s.label }))}
        chosen={sweepKey}
        onChoose={setSweepKey}
        rows={rows}
        outputLabel="Split"
        valueLabel="Win probability"
      />

      <Controls>
        <Field label="Your budget" value={v.budget} onChange={(x) => set({ budget: Math.round(x) })} min={1} max={12} step={1} format={(x) => `${Math.round(x)} units`} />
        <Field label="Rival budget" value={v.rivalBudget} onChange={(x) => set({ rivalBudget: Math.round(x) })} min={1} max={12} step={1} format={(x) => `${Math.round(x)} units`} />
        <Field label="Value of account A" value={v.a} onChange={(x) => set({ a: x })} min={0.1} max={5} step={0.1} format={(x) => num(x, 1)} />
        <Field label="Value of account B" value={v.b} onChange={(x) => set({ b: x })} min={0.1} max={5} step={0.1} format={(x) => num(x, 1)} />
        <Field label="Value of account C" value={v.c} onChange={(x) => set({ c: x })} min={0.1} max={5} step={0.1} format={(x) => num(x, 1)} />
        <Field
          label="Precision"
          value={v.precision}
          onChange={(x) => set({ precision: x })}
          min={0.1}
          max={10}
          step={0.1}
          format={(x) => num(x, 1)}
          help="Higher means the rival concentrates on their best splits; lower spreads them evenly."
        />
      </Controls>
    </div>
  );
}
