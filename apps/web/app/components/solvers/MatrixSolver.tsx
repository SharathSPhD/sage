"use client";

/* A payoff table you write yourself, solved by the engine.
 *
 * This is the standards / adoption case: three moves a side, payoffs built
 * from installed base, switching cost and the gain from agreeing. The mix
 * comes back from POST /v1/solve/qre; the expected value of each move is that
 * mix against the table you specified.
 */

import { useMemo, useState } from "react";
import { useSolve, type QRESolution } from "../../../lib/problems";
import { PayoffMatrix } from "../charts/PayoffMatrix";
import { Answer, Bars, Controls, Field, Figure, ModelLine, num } from "./ui";

interface Inputs {
  agreement: number;
  yourBase: number;
  theirBase: number;
  switchCost: number;
  goneAlone: number;
  precision: number;
}

const DEFAULTS: Inputs = {
  agreement: 10,
  yourBase: 7,
  theirBase: 6,
  switchCost: 3,
  goneAlone: 2,
  precision: 0.4,
};

const YOURS = ["Back your format", "Adopt theirs", "Go proprietary"];
const THEIRS = ["Back their format", "Adopt yours", "Go proprietary"];

/** 0 = back your format, 1 = adopt theirs, 2 = go proprietary. */
function table(v: Inputs): { u1: number[][]; u2: number[][] } {
  const u1: number[][] = [];
  const u2: number[][] = [];
  for (let i = 0; i < 3; i++) {
    u1.push([]);
    u2.push([]);
    for (let j = 0; j < 3; j++) {
      const bothOnYours = i === 0 && j === 1;
      const bothOnTheirs = i === 1 && j === 0;
      let a: number;
      let b: number;
      if (bothOnYours) {
        a = v.yourBase + v.agreement;
        b = v.agreement - v.switchCost;
      } else if (bothOnTheirs) {
        a = v.agreement - v.switchCost;
        b = v.theirBase + v.agreement;
      } else {
        a = i === 2 ? v.goneAlone : i === 0 ? v.yourBase : -v.switchCost;
        b = j === 2 ? v.goneAlone : j === 0 ? v.theirBase : -v.switchCost;
      }
      u1[i].push(a);
      u2[i].push(b);
    }
  }
  return { u1, u2 };
}

const pts = (x: number) => `${x >= 0 ? "" : "−"}${Math.abs(x).toFixed(1)} pts`;

export function MatrixSolver() {
  const [v, setV] = useState<Inputs>(DEFAULTS);
  const set = (patch: Partial<Inputs>) => setV((old) => ({ ...old, ...patch }));

  const { u1, u2 } = useMemo(() => table(v), [v]);
  const body = useMemo(() => ({ payoffs: [u1, u2], lam: v.precision }), [u1, u2, v.precision]);
  const { data, error, busy } = useSolve<QRESolution>("qre", body);

  const values = data ? u1.map((row) => row.reduce((acc, x, j) => acc + x * data.sigma[1][j], 0)) : null;
  const best = values ? values.indexOf(Math.max(...values)) : -1;
  const ranked = values
    ? values.map((value, i) => ({ i, value })).sort((a, b) => b.value - a.value)
    : [];

  return (
    <div className="studio">
      <Answer headline={best >= 0 ? `${YOURS[best]}.` : "Solving…"} busy={busy} error={error}>
        {values && (
          <>
            <Figure label="Expected value" value={pts(values[best])} tween={values[best]} format={pts} note="share points of the category per year" />
            <Figure
              label="Next best gives up"
              value={pts(ranked[0].value - ranked[1].value)}
              note={YOURS[ranked[1].i].toLowerCase()}
              tone="neutral"
            />
            <Figure
              label="Their most likely move"
              value={data ? THEIRS[data.sigma[1].indexOf(Math.max(...data.sigma[1]))] : "—"}
              note={data ? `${(100 * Math.max(...data.sigma[1])).toFixed(0)}% of the weight` : ""}
              tone="neutral"
            />
          </>
        )}
      </Answer>

      {data && (
        <ModelLine>
          3 x 3 payoff table, both sides solved together, precision {num(v.precision, 2)}. Residual{" "}
          {data.residual.toExponential(1)} in {data.n_iter} iterations.
        </ModelLine>
      )}

      {data && (
        <section className="card">
          <h3>The table, and where the two of you land on it</h3>
          <PayoffMatrix
            rowLabels={YOURS}
            colLabels={THEIRS}
            payoffs={u1.map((row, i) => row.map((a, j) => [a, u2[i][j]] as [number, number]))}
            rowMix={data.sigma[0]}
            colMix={data.sigma[1]}
            format={(x) => x.toFixed(1)}
          />
        </section>
      )}

      <div className="answer-cols">
        <section className="card">
          <h3>What each move is worth</h3>
          {values && (
            <table className="alt-table">
              <thead>
                <tr>
                  <th scope="col">Move</th>
                  <th scope="col">Expected value</th>
                  <th scope="col">Gives up</th>
                </tr>
              </thead>
              <tbody>
                {ranked.map((r) => (
                  <tr key={r.i} data-us={r.i === best}>
                    <th scope="row">{YOURS[r.i]}</th>
                    <td>{pts(r.value)}</td>
                    <td className="alt-cost">{r.i === best ? "—" : `−${pts(values[best] - r.value)}`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
        <section className="card">
          <h3>What they are likely to do</h3>
          {data && <Bars rows={data.sigma[1].map((p, j) => ({ label: THEIRS[j], p }))} tone="rival" />}
        </section>
      </div>

      <Controls>
        <Field label="Extra market if you both land on one format" value={v.agreement} onChange={(x) => set({ agreement: x })} min={0} max={25} step={0.5} format={(x) => num(x, 1)} />
        <Field label="Your installed base on your format" value={v.yourBase} onChange={(x) => set({ yourBase: x })} min={0} max={20} step={0.5} format={(x) => num(x, 1)} />
        <Field label="Their installed base on theirs" value={v.theirBase} onChange={(x) => set({ theirBase: x })} min={0} max={20} step={0.5} format={(x) => num(x, 1)} />
        <Field label="Cost of moving off your own format" value={v.switchCost} onChange={(x) => set({ switchCost: x })} min={0} max={15} step={0.5} format={(x) => num(x, 1)} />
        <Field label="What a private format is worth alone" value={v.goneAlone} onChange={(x) => set({ goneAlone: x })} min={0} max={15} step={0.5} format={(x) => num(x, 1)} />
        <Field
          label="Precision"
          value={v.precision}
          onChange={(x) => set({ precision: x })}
          min={0.02}
          max={4}
          step={0.02}
          format={(x) => num(x, 2)}
          help="Per share point. Higher means the other side commits more decisively to its best move."
        />
      </Controls>
    </div>
  );
}
