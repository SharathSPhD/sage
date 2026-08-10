"use client";

import { useCallback, useState } from "react";
import { solveQRE, type TwoPlayerGame } from "../../../lib/qre";
import { Bars, LambdaSlider, PanelShell } from "./ui";

/* "You predict it" (Distill pattern) for the λ-estimator family: the panel
   draws synthetic choice data at a hidden λ*, you read the frequencies and
   guess; the live estimator family then shows what the instruments say —
   and then the truth. Estimation runs on the deployed solver. */

const GAME: TwoPlayerGame = {
  u1: [
    [3, 0, 1.5],
    [1, 2, 0.5],
    [0, 1, 2.5],
  ],
  u2: [
    [2, 1, 0],
    [0.5, 3, 1],
    [1.5, 0, 2],
  ],
};
const N = 20000;
const ACTIONS = ["a1", "a2", "a3"];

interface Estimates {
  mle: { lam: number; ci_low: number; ci_high: number };
  dispersion: { lam: number };
  warnings: string[];
}

function drawCounts(sigma: number[], n: number): number[] {
  const counts = sigma.map(() => 0);
  for (let i = 0; i < n; i++) {
    let r = Math.random();
    let k = 0;
    while (k < sigma.length - 1 && r > sigma[k]) {
      r -= sigma[k];
      k++;
    }
    counts[k]++;
  }
  return counts;
}

export function GuessLambda() {
  const [truth, setTruth] = useState<number | null>(null);
  const [counts, setCounts] = useState<number[][] | null>(null);
  const [guess, setGuess] = useState(1.0);
  const [revealed, setRevealed] = useState(false);
  const [estimates, setEstimates] = useState<Estimates | null>(null);
  const [busy, setBusy] = useState(false);

  const deal = useCallback(() => {
    const lamStar = 0.3 * Math.pow(15, Math.random()); // log-uniform 0.3..4.5
    const q = solveQRE(GAME, lamStar);
    setTruth(lamStar);
    setCounts([drawCounts(q.sigma1, N), drawCounts(q.sigma2, N)]);
    setRevealed(false);
    setEstimates(null);
  }, []);

  const reveal = useCallback(() => {
    if (!counts) return;
    setBusy(true);
    fetch("/api/v1/estimate/lambda", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ payoffs: [GAME.u1, GAME.u2], counts }),
    })
      .then((r) => r.json())
      .then((d) => {
        setEstimates({
          mle: d.estimates.mle,
          dispersion: d.estimates.dispersion,
          warnings: d.warnings ?? [],
        });
        setRevealed(true);
        setBusy(false);
      })
      .catch(() => setBusy(false));
  }, [counts]);

  return (
    <PanelShell title="guess λ from the data — then let the instruments answer" provenance="live">
      {!counts ? (
        <button data-primary="true" onClick={deal}>
          deal a hidden λ and draw {N.toLocaleString()} choices
        </button>
      ) : (
        <div className="panel-cols">
          <div>
            <div className="panel-label">observed choice frequencies (n = {N.toLocaleString()} per player)</div>
            {counts.map((c, p) => {
              const tot = c.reduce((a, b) => a + b, 0);
              return (
                <div key={p} style={{ marginBottom: "0.6rem" }}>
                  <Bars
                    values={c.map((x) => x / tot)}
                    labels={ACTIONS.map((a) => `P${p + 1} ${a}`)}
                    max={1}
                    format={(v) => v.toFixed(3)}
                  />
                </div>
              );
            })}
            <LambdaSlider lam={guess} setLam={setGuess} min={0.1} max={10} label="your guess for λ" />
            <div style={{ display: "flex", gap: "0.6rem", marginTop: "0.6rem" }}>
              <button data-primary="true" onClick={reveal} disabled={busy}>
                {busy ? "estimating…" : "reveal — run the estimators"}
              </button>
              <button onClick={deal}>new deal</button>
            </div>
          </div>
          <div>
            {revealed && estimates && truth !== null ? (
              <div>
                <div className="panel-label">verdicts</div>
                <table style={{ width: "100%", fontFamily: "var(--mono)", fontSize: "0.88rem" }}>
                  <tbody>
                    <tr>
                      <td style={{ color: "var(--text-faint)", padding: "0.25rem 0" }}>your guess</td>
                      <td style={{ textAlign: "right" }}>{guess.toFixed(2)}</td>
                    </tr>
                    <tr>
                      <td style={{ color: "var(--text-faint)", padding: "0.25rem 0" }}>frequency MLE</td>
                      <td style={{ textAlign: "right", color: "var(--accent)" }}>
                        {estimates.mle.lam.toFixed(2)}{" "}
                        <span style={{ color: "var(--text-faint)", fontSize: "0.75rem" }}>
                          [{estimates.mle.ci_low.toFixed(2)}, {estimates.mle.ci_high.toFixed(2)}]
                        </span>
                      </td>
                    </tr>
                    <tr>
                      <td style={{ color: "var(--text-faint)", padding: "0.25rem 0" }}>dispersion inversion</td>
                      <td style={{ textAlign: "right", color: "var(--accent)" }}>{estimates.dispersion.lam.toFixed(2)}</td>
                    </tr>
                    <tr style={{ borderTop: "1px solid var(--border)" }}>
                      <td style={{ padding: "0.25rem 0" }}>hidden truth λ*</td>
                      <td style={{ textAlign: "right", color: "var(--amber)", fontWeight: 700 }}>{truth.toFixed(2)}</td>
                    </tr>
                  </tbody>
                </table>
                {estimates.warnings.length > 0 && (
                  <div className="warnings" style={{ marginTop: "0.6rem" }}>
                    {estimates.warnings.map((w) => (
                      <div className="w" key={w}>
                        ⚠ {w}
                      </div>
                    ))}
                  </div>
                )}
                <p style={{ fontSize: "0.78rem", color: "var(--text-faint)", marginTop: "0.6rem" }}>
                  Two structurally different estimators, one answer — when the data really are a
                  single-λ QRE. On real data their <em>disagreement</em> is the finding.
                </p>
              </div>
            ) : (
              <p style={{ color: "var(--text-faint)", fontSize: "0.85rem" }}>
                Sharper frequencies ⇒ higher λ. Commit to a guess before revealing — the point of
                the exercise is calibrating your own eye against the instruments.
              </p>
            )}
          </div>
        </div>
      )}
    </PanelShell>
  );
}
