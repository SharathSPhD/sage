"use client";

import { useState } from "react";
import { softmax } from "../../../lib/qre";
import { Bars, LambdaSlider, NumberDial, PanelShell } from "./ui";

/* Doc 08's promised panel: two dials people conflate, shown orthogonal.
   Elasticity reshapes the payoff surface itself (which prices are good
   ideas); λ only sharpens how decisively the best one is chosen. */

const PRICES = [1.7, 1.72, 1.74, 1.76, 1.78];
const COST = 1.6;

function profits(elasticity: number): number[] {
  // simple monopoly profit: (p - c) · D(p), D(p) = 100·exp(−ε·(p − p₀))
  return PRICES.map((p) => (p - COST) * 100 * Math.exp(-elasticity * (p - PRICES[0])));
}

export function TwoDials() {
  const [elasticity, setElasticity] = useState(18);
  const [lam, setLam] = useState(0.5);
  const u = profits(elasticity);
  const p = softmax(u, lam);
  const best = u.indexOf(Math.max(...u));

  return (
    <PanelShell title="elasticity moves the payoffs; λ moves the sharpness" provenance="client">
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.4rem", alignItems: "start" }}>
        <div>
          <NumberDial
            value={elasticity}
            setValue={setElasticity}
            min={2}
            max={40}
            step={0.5}
            label="demand elasticity ε (consumers)"
            format={(v) => v.toFixed(1)}
          />
          <div style={{ marginTop: "0.8rem" }}>
            <div className="panel-label">
              expected profit — the payoff surface itself{" "}
              <span className="badge">best: £{PRICES[best].toFixed(2)}</span>
            </div>
            <Bars values={u} labels={PRICES.map((x) => `£${x.toFixed(2)}`)} format={(v) => v.toFixed(1)} />
          </div>
        </div>
        <div>
          <LambdaSlider lam={lam} setLam={setLam} min={0.02} max={8} label="response precision λ (the firm)" />
          <div style={{ marginTop: "0.8rem" }}>
            <div className="panel-label">choice probabilities — only the sharpness</div>
            <Bars values={p} labels={PRICES.map((x) => `£${x.toFixed(2)}`)} max={1} format={(v) => v.toFixed(3)} />
          </div>
        </div>
      </div>
      <p style={{ fontSize: "0.78rem", color: "var(--text-faint)", marginTop: "0.8rem" }}>
        Turn ε: the payoff bars move, the best price can change, and the probabilities follow —
        that is a fact about <em>consumers</em>. Turn λ with ε frozen: the bars never move; only
        how decisively the best is chosen changes — a fact about <em>the firm</em>. Two dials,
        orthogonal by construction; conflating them is how mis-specified demand leaks into λ.
      </p>
    </PanelShell>
  );
}
