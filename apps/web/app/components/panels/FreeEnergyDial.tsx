"use client";

import { useState } from "react";
import { entropy, softmax } from "../../../lib/qre";
import { Bars, NumberDial, PanelShell } from "./ui";

/* Doc 04's promised panel: slide the 1/λ (temperature) dial and watch the
   optimum morph from uniform (entropy dominates) to argmax (payoff
   dominates), with the free-energy decomposition E[U] + T·H shown live. */

const U = [102, 108, 115, 113, 104];
const LABELS = ["a", "b", "c", "d", "e"];

export function FreeEnergyDial() {
  const [temp, setTemp] = useState(5); // T = 1/λ
  const lam = 1 / Math.max(temp, 1e-6);
  const p = softmax(U, lam);
  const eu = p.reduce((acc, v, i) => acc + v * U[i], 0);
  const h = entropy(p);
  const objective = eu + temp * h;

  return (
    <PanelShell title="the Gibbs variational principle" provenance="client">
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.4rem", alignItems: "start" }}>
        <div>
          <NumberDial
            value={temp}
            setValue={setTemp}
            min={0.05}
            max={30}
            step={0.05}
            label="temperature T = 1/λ"
          />
          <div style={{ marginTop: "1rem" }}>
            <div className="panel-label">the optimum σ* = argmax E[U] + T·H(σ)</div>
            <Bars values={p} labels={LABELS} max={1} format={(v) => v.toFixed(3)} />
          </div>
        </div>
        <div>
          <div className="panel-label">free-energy decomposition</div>
          <table style={{ width: "100%", fontFamily: "var(--mono)", fontSize: "0.85rem" }}>
            <tbody>
              <tr>
                <td style={{ color: "var(--text-faint)", padding: "0.25rem 0" }}>E[U] · payoff</td>
                <td style={{ textAlign: "right", color: "var(--accent)" }}>{eu.toFixed(2)}</td>
              </tr>
              <tr>
                <td style={{ color: "var(--text-faint)", padding: "0.25rem 0" }}>T·H(σ) · option value</td>
                <td style={{ textAlign: "right", color: "var(--amber)" }}>{(temp * h).toFixed(2)}</td>
              </tr>
              <tr style={{ borderTop: "1px solid var(--border)" }}>
                <td style={{ padding: "0.25rem 0" }}>objective (−free energy)</td>
                <td style={{ textAlign: "right" }}>{objective.toFixed(2)}</td>
              </tr>
            </tbody>
          </table>
          <p style={{ fontSize: "0.78rem", color: "var(--text-faint)", marginTop: "0.8rem" }}>
            Hot (T large): the entropy bonus dominates and the optimum is nearly uniform. Cold
            (T → 0): payoff dominates and the optimum is the argmax. The softmax is not an
            assumption — it is <em>the</em> solution of this trade-off at every temperature.
            Physicists: F = U − TS, verbatim, payoff as negative energy.
          </p>
        </div>
      </div>
    </PanelShell>
  );
}
