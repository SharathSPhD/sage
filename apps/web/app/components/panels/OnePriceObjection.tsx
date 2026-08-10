"use client";

import { useMemo, useState } from "react";
import { argmaxMix, expectedPayoffsRow, solveQRE, type TwoPlayerGame } from "../../../lib/qre";
import { Bars, LambdaSlider, PanelShell } from "./ui";

/* Doc 09's promised panel: the same market shown twice — "argmax against a
   point rival" vs "argmax against the QRE rival distribution" — with the
   profit difference and its sensitivity to the rival assumption. */

const PRICES = ["£1.70", "£1.72", "£1.74", "£1.76", "£1.78"];

// A 5-price Bertrand-ish duopoly: undercutting steals demand; margins rise
// with price. u[i][j] = my profit at price i when rival prices j.
function duopoly(): TwoPlayerGame {
  const margin = [10, 12, 14, 16, 18];
  const u = (i: number, j: number) => {
    const share = i < j ? 0.75 : i === j ? 0.5 : 0.25;
    return margin[i] * share * 10;
  };
  const u1 = PRICES.map((_, i) => PRICES.map((_, j) => u(i, j)));
  const u2 = PRICES.map((_, i) => PRICES.map((_, j) => u(j, i)));
  return { u1, u2 };
}

export function OnePriceObjection() {
  const [rivalLam, setRivalLam] = useState(1.0);
  const game = useMemo(duopoly, []);

  const { pointEU, qreEU, rivalMix, pointRival } = useMemo(() => {
    const qre = solveQRE(game, rivalLam);
    const rivalMix = qre.sigma2;
    // point rival: their argmax against MY QRE mix (a best-responding rival)
    const rivalPayoffs = game.u2.reduce(
      (acc, row, i) => acc.map((v, j) => v + row[j] * qre.sigma1[i]),
      new Array(PRICES.length).fill(0),
    );
    const pointRival = argmaxMix(rivalPayoffs);
    return {
      rivalMix,
      pointRival,
      pointEU: expectedPayoffsRow(game.u1, pointRival),
      qreEU: expectedPayoffsRow(game.u1, rivalMix),
    };
  }, [game, rivalLam]);

  const pointBest = pointEU.indexOf(Math.max(...pointEU));
  const qreBest = qreEU.indexOf(Math.max(...qreEU));
  // regret of the point-rival recommendation, evaluated under the QRE rival
  const regret = qreEU[qreBest] - qreEU[pointBest];

  return (
    <PanelShell title="argmax against a point rival vs the QRE distribution" provenance="client">
      <LambdaSlider lam={rivalLam} setLam={setRivalLam} min={0.1} max={20} label="how noisy is the rival? (their λ)" />
      <div className="panel-cols" style={{ marginTop: "1rem" }}>
        <div>
          <div className="panel-label">
            rival as a POINT prediction <span className="badge">recommend {PRICES[pointBest]}</span>
          </div>
          <Bars values={pointRival} labels={PRICES} max={1} format={(v) => v.toFixed(2)} color="var(--text-faint)" />
          <div className="panel-label" style={{ marginTop: "0.7rem" }}>my expected profit per price</div>
          <Bars values={pointEU} labels={PRICES} format={(v) => v.toFixed(1)} color="var(--blue)" />
        </div>
        <div>
          <div className="panel-label">
            rival as the QRE DISTRIBUTION <span className="badge" data-tone="ok">recommend {PRICES[qreBest]}</span>
          </div>
          <Bars values={rivalMix} labels={PRICES} max={1} format={(v) => v.toFixed(2)} color="var(--amber)" />
          <div className="panel-label" style={{ marginTop: "0.7rem" }}>my expected profit per price</div>
          <Bars values={qreEU} labels={PRICES} format={(v) => v.toFixed(1)} />
        </div>
      </div>
      <div style={{ marginTop: "1rem" }}>
        <span className="reading" data-tone={regret > 0.5 ? "warn" : "neutral"} style={{ fontSize: "1.15rem" }}>
          cost of the point assumption: {regret.toFixed(2)} profit
        </span>
        <p style={{ fontSize: "0.78rem", color: "var(--text-faint)", marginTop: "0.4rem" }}>
          Both sides take an argmax — the objection is right that you charge ONE price. What
          differs is the rival model the argmax is taken against. Slide the rival&apos;s λ: when
          they are noisy, the point prediction is badly wrong and the distribution earns its
          keep; as λ → ∞ the two recommendations converge and the objection wins on its own
          terms. The distribution is the deliverable.
        </p>
      </div>
    </PanelShell>
  );
}
