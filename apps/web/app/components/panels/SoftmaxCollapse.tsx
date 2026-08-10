"use client";

import { useState } from "react";
import { argmaxMix, softmax } from "../../../lib/qre";
import { Bars, LambdaSlider, PanelShell } from "./ui";

/* Doc 01's promised panel: payoff bars → probability bars; λ slider from 0
   to ∞; watch the collapse to Nash. Payoffs are editable. */

const PRICES = ["£1.70", "£1.72", "£1.74", "£1.76", "£1.78"];
const DEFAULT = [102, 108, 115, 113, 104];

export function SoftmaxCollapse() {
  const [payoffs, setPayoffs] = useState<number[]>(DEFAULT);
  const [lam, setLam] = useState<number>(0.2);
  const probs = lam === Infinity ? argmaxMix(payoffs) : softmax(payoffs, lam);
  const collapsed = Math.max(...probs) > 0.99;

  return (
    <PanelShell title="softmax and λ" provenance="client">
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.4rem", alignItems: "start" }}>
        <div>
          <div className="panel-label">expected profit per price — drag to edit</div>
          {payoffs.map((v, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
              <span className="mono" style={{ width: "3.4rem", fontSize: "0.72rem", color: "var(--text-faint)" }}>
                {PRICES[i]}
              </span>
              <input
                type="range"
                min={80}
                max={130}
                step={1}
                value={v}
                onChange={(e) =>
                  setPayoffs(payoffs.map((p, j) => (j === i ? Number(e.target.value) : p)))
                }
                style={{ flex: 1 }}
                aria-label={`profit at ${PRICES[i]}`}
              />
              <span className="mono" style={{ width: "2.2rem", fontSize: "0.72rem", color: "var(--text-dim)" }}>
                {v}
              </span>
            </div>
          ))}
          <div style={{ marginTop: "1rem" }}>
            <LambdaSlider lam={lam} setLam={setLam} min={0.01} max={10} infinity />
          </div>
        </div>
        <div>
          <div className="panel-label">
            choice probabilities{" "}
            {collapsed ? (
              <span className="badge" data-tone="warn">
                collapsed to Nash
              </span>
            ) : null}
          </div>
          <Bars values={probs} labels={PRICES} max={1} format={(v) => v.toFixed(3)} />
          <p style={{ fontSize: "0.78rem", color: "var(--text-faint)", marginTop: "0.8rem" }}>
            At λ ≈ 0 every price is equally likely; slide right and probability piles onto the
            argmax. Notice the 2-points-behind price keeps real probability long after the
            13-points-behind one is gone — <em>magnitude matters</em>, which is exactly what Nash
            throws away.
          </p>
        </div>
      </div>
    </PanelShell>
  );
}
