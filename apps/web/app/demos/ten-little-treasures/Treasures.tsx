"use client";

import { useMemo, useState } from "react";
import { solve2x2 } from "../../../lib/demos/gametheory";
import { Readout, Widget } from "../components/chrome";
import { useDragNumber } from "../components/drag";

/* Goeree, J. K. and Holt, C. A. (2001), "Ten Little Treasures of Game Theory and
 * Ten Intuitive Contradictions", American Economic Review 91(5), 1402-1422.
 * Payoffs in US dollars, as printed. 50 subjects in five cohorts of ten, one-shot
 * with random pairing, each treatment played once. Numbers below are the
 * published ones, checked against the paper text before being typed here. */

const COLUMN_PAYOFFS = [
  [0.4, 0.8],
  [0.8, 0.4],
];

interface Treatment {
  id: "symmetric" | "high" | "low";
  name: string;
  rowPayoffs: number[][];
  dataTop: number;
  dataLeft: number;
  nashTop: number;
  nashLeft: number;
}

const TREATMENTS: Treatment[] = [
  {
    id: "symmetric",
    name: "the basic game",
    rowPayoffs: [
      [0.8, 0.4],
      [0.4, 0.8],
    ],
    dataTop: 0.48,
    dataLeft: 0.48,
    nashTop: 0.5,
    nashLeft: 0.5,
  },
  {
    id: "high",
    name: "Top-Left raised to $3.20",
    rowPayoffs: [
      [3.2, 0.4],
      [0.4, 0.8],
    ],
    dataTop: 0.96,
    dataLeft: 0.16,
    nashTop: 0.5,
    nashLeft: 0.125,
  },
  {
    id: "low",
    name: "Top-Left cut to $0.44",
    rowPayoffs: [
      [0.44, 0.4],
      [0.4, 0.8],
    ],
    dataTop: 0.08,
    dataLeft: 0.8,
    nashTop: 0.5,
    nashLeft: 0.9090909090909091,
  },
];

const HIGH = TREATMENTS[1];
const LOW = TREATMENTS[2];
const BASE = TREATMENTS[0];

export function Treasures() {
  const [guess, setGuess] = useState(0.5);
  const [locked, setLocked] = useState(false);
  const [lam, setLam] = useState(3);

  const curves = useMemo(() => {
    const grid = Array.from({ length: 121 }, (_, i) => 0.05 * Math.pow(60 / 0.05, i / 120));
    return {
      grid,
      series: TREATMENTS.map((t) => grid.map((l) => solve2x2(t.rowPayoffs, COLUMN_PAYOFFS, l)[0])),
    };
  }, []);

  const atLam = useMemo(() => TREATMENTS.map((t) => solve2x2(t.rowPayoffs, COLUMN_PAYOFFS, lam)[0]), [lam]);

  const bestLam = useMemo(() => {
    let best = 0.05;
    let sse = Infinity;
    for (let i = 0; i <= 2000; i++) {
      const l = 0.05 * Math.pow(60 / 0.05, i / 2000);
      let s = 0;
      for (const t of TREATMENTS) s += (solve2x2(t.rowPayoffs, COLUMN_PAYOFFS, l)[0] - t.dataTop) ** 2;
      if (s < sse) {
        sse = s;
        best = l;
      }
    }
    return best;
  }, []);

  const gapClosed = (atLam[1] - 0.5) / (HIGH.dataTop - 0.5);

  return (
    <>
      <Widget
        hook="One payoff changes. Nash does not move."
        lede={
          <>
            <p>
              A two-by-two game. The Row player&apos;s mixing probability in Nash equilibrium is fixed entirely by the{" "}
              <em>Column</em> player&apos;s payoffs — and those are identical in all three versions below. So Nash says
              Row plays Top exactly half the time, whatever happens to Row&apos;s own numbers.
            </p>
          </>
        }
        consequence={
          <>
            In the basic game the subjects obliged: 48% chose Top against a Nash prediction of 50%. That is the treasure.
            The contradiction is on the next widget.
          </>
        }
        maths={
          <>
            <p>
              Row&apos;s equilibrium probability p solves Column&apos;s indifference: p(0.40 − 0.80) + (1 − p)(0.80 −
              0.40) = 0, so p = 0.5 in every treatment, because Column&apos;s four payoffs never change. Column&apos;s
              equilibrium probability q solves Row&apos;s indifference and therefore <em>does</em> move: q = 0.5, 0.125
              and 0.909 in the three treatments.
            </p>
            <p>
              Source: Goeree &amp; Holt (2001),{" "}
              <a href="https://www.aeaweb.org/articles?id=10.1257/aer.91.5.1402" rel="noreferrer">
                &ldquo;Ten Little Treasures of Game Theory and Ten Intuitive Contradictions&rdquo;
              </a>
              , <em>American Economic Review</em> 91(5), 1402–1422, matching-pennies treatments. 50 subjects in five
              cohorts of ten, one-shot with random pairing. Every experimental number on this page is from that paper;
              none is simulated, illustrative or reconstructed.
            </p>
          </>
        }
      >
        <PayoffTable treatment={BASE} />
      </Widget>

      <Widget
        hook="Guess what the subjects did"
        lede={
          <p>
            Now Row&apos;s payoff in the top-left cell is raised from $0.80 to $3.20. Column&apos;s payoffs are
            untouched, so the Nash prediction for Row is still exactly 50%. Drag the bar to where you think the actual
            subjects landed, then lock it in.
          </p>
        }
        consequence={
          locked ? (
            <>
              96% chose Top. Nash could not move — Row&apos;s own payoff does not enter Row&apos;s equilibrium condition
              — and behaviour moved almost the whole way. Cut the same cell to $0.44 instead and Top falls to 8%.
            </>
          ) : (
            <>Your guess is not scored against theory. It is scored against 50 people who actually played this game.</>
          )
        }
        maths={
          <>
            <p>
              Observed Row choices: 48% Top in the basic game, 96% Top with the $3.20 cell, 8% Top with the $0.44 cell.
              Column choices moved in the direction Nash predicts for Column — 48%, 16%, 80% Left against Nash values of
              50%, 12.5%, 90.9% — which is why the paper calls the Row result an own-payoff effect rather than
              confusion.
            </p>
            <p>
              Goeree &amp; Holt (2001), AER 91(5), 1402–1422. n = 50 subjects, one-shot, random pairing, payoffs in
              dollars exactly as printed above.
            </p>
          </>
        }
      >
        <GuessPanel guess={guess} setGuess={setGuess} locked={locked} />
        <div className="treasure-actions">
          <PayoffTable treatment={HIGH} highlight />
          {!locked ? (
            <button type="button" className="btn" data-primary="true" onClick={() => setLocked(true)}>
              Lock in {(guess * 100).toFixed(0)}% and show the data
            </button>
          ) : (
            <button type="button" className="btn" onClick={() => setLocked(false)}>
              Hide the answer and guess again
            </button>
          )}
        </div>
      </Widget>

      <Widget
        hook="One dial moves the prediction Nash cannot move"
        lede={
          <p>
            Give the players a precision λ instead of perfect maximisation and the equilibrium condition changes shape:
            Row&apos;s own payoffs now enter Row&apos;s own behaviour. Drag the λ line and watch the three predictions
            separate from the flat Nash line at 50%.
          </p>
        }
        consequence={
          <>
            At λ = {lam.toFixed(2)} the model puts Row on Top {(atLam[1] * 100).toFixed(0)}% of the time in the $3.20
            treatment, closing {(gapClosed * 100).toFixed(0)}% of the distance from Nash to the data. Nash closes none of
            it, at any parameter value, because it has none to spend.
          </>
        }
        maths={
          <>
            <p>
              The curves are the logit equilibrium of each 2×2 game, solved by bisection on the column player&apos;s
              probability rather than by damped iteration, because matching pennies cycles under iteration at every λ
              worth plotting. Payoffs are in dollars, so λ carries units of 1/dollar.
            </p>
            <p>
              Least squares over the three Row frequencies puts the best fit at λ = {bestLam.toFixed(2)}, giving
              predicted Top of{" "}
              {TREATMENTS.map((t) => (solve2x2(t.rowPayoffs, COLUMN_PAYOFFS, bestLam)[0] * 100).toFixed(0)).join("% / ")}%
              against observed 48% / 96% / 8%. Logit alone therefore captures the direction and roughly two-thirds of
              the $3.20 effect and about a third of the $0.44 effect — it does not land on the data. Goeree, Holt and
              Palfrey added risk aversion to close the rest (<em>Games and Economic Behavior</em>, 2003); this page shows
              the plain logit fit and does not pretend otherwise.
            </p>
            <p>
              The peak of the $3.20 curve is 0.847 at λ ≈ 3.03, and every curve returns to 0.5 as λ → ∞ because that is
              the Nash point. The own-payoff effect is a property of the interior of the λ range, not of its limit.
            </p>
          </>
        }
      >
        <LambdaChart curves={curves} lam={lam} setLam={setLam} bestLam={bestLam} />
      </Widget>

      <Widget
        hook="The same shape, in a different game"
        lede={
          <p>
            In the traveler&apos;s dilemma two players claim an integer between 180 and 300; the lower claim is paid to
            both, with a reward R added to the low claimer and taken from the high one. Iterated deletion leaves 180 as
            the only Nash equilibrium — for every R.
          </p>
        }
        consequence={
          <>
            With R = 180 the subjects played the Nash equilibrium: average claim 201. With R = 5 — same game, same
            equilibrium — the average claim was 280, at the opposite end of the range.
          </>
        }
        maths={
          <>
            <p>
              Goeree &amp; Holt (2001), AER 91(5), 1402–1422, traveler&apos;s dilemma treatments: 50 subjects (25 pairs)
              played R = 180 and then a matched R = 5 treatment. &ldquo;Close to 80 percent of all the subjects chose the
              Nash equilibrium strategy, with an average claim of 201&rdquo;; in the low-R treatment &ldquo;roughly the
              same fraction chose the highest possible claim … for which the average was 280&rdquo;.
            </p>
            <p>
              No model with a parameter is fitted to these two numbers here. They are on the page because they are the
              cleanest statement of the pattern: the equilibrium is a fixed point of the payoff structure, and behaviour
              is a function of the payoff <em>gradients</em> the structure leaves behind.
            </p>
          </>
        }
      >
        <TravellerBars />
      </Widget>
    </>
  );
}

function PayoffTable({ treatment, highlight }: { treatment: Treatment; highlight?: boolean }) {
  return (
    <table className="payoff-table treasure-table">
      <caption className="panel-label">{treatment.name} — Row, Column payoffs in dollars</caption>
      <thead>
        <tr>
          <th scope="col">
            <span className="visually-hidden">Row action</span>
          </th>
          <th scope="col">Left</th>
          <th scope="col">Right</th>
        </tr>
      </thead>
      <tbody>
        {["Top", "Bottom"].map((label, i) => (
          <tr key={label}>
            <th scope="row">{label}</th>
            {[0, 1].map((j) => (
              <td key={j} data-changed={highlight && i === 0 && j === 0 ? "true" : undefined}>
                {treatment.rowPayoffs[i][j].toFixed(2)}, {COLUMN_PAYOFFS[i][j].toFixed(2)}
                {highlight && i === 0 && j === 0 ? <span className="visually-hidden"> (the changed cell)</span> : null}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function GuessPanel({ guess, setGuess, locked }: { guess: number; setGuess: (v: number) => void; locked: boolean }) {
  const W = 460;
  const H = 250;
  const base = 200;
  const top = 30;
  const Y = (p: number) => base - (base - top) * p;
  const drag = useDragNumber({
    value: guess,
    min: 0,
    max: 1,
    step: 0.01,
    onChange: setGuess,
    axis: "y",
    travelPx: 170,
    label: "Your guess: the share of subjects choosing Top",
    valueText: (v) => `${(v * 100).toFixed(0)} percent`,
  });
  const bars: { x: number; p: number; label: string; fill: string; pattern?: boolean }[] = [
    { x: 60, p: guess, label: "your guess", fill: "var(--accent-strong)" },
    { x: 190, p: 0.5, label: "Nash", fill: "var(--text-3)" },
    { x: 320, p: HIGH.dataTop, label: "the data", fill: "var(--q-whirlpool)", pattern: true },
  ];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="demo-svg demo-svg-guess" role="img" aria-label={`Share choosing Top. Your guess ${(guess * 100).toFixed(0)} percent${locked ? `, Nash 50 percent, observed 96 percent` : ""}.`}>
      <defs>
        <pattern id="data-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="6" height="6" fill="var(--q-whirlpool)" />
          <line x1="0" y1="0" x2="0" y2="6" stroke="var(--surface)" strokeWidth="2" />
        </pattern>
      </defs>
      <line x1={40} y1={base} x2={W - 20} y2={base} stroke="var(--text-3)" />
      {[0, 0.5, 1].map((g) => (
        <g key={g}>
          <line x1={40} y1={Y(g)} x2={W - 20} y2={Y(g)} stroke="var(--border)" strokeDasharray="2 4" />
          <text x={34} y={Y(g) + 4} textAnchor="end" fontSize={11} fill="var(--text-3)" fontFamily="var(--mono)">
            {g * 100}%
          </text>
        </g>
      ))}
      {bars.map((b) => {
        if ((b.label === "Nash" || b.label === "the data") && !locked) {
          return (
            <text key={b.label} x={b.x + 34} y={base + 20} textAnchor="middle" fontSize={11} fill="var(--text-3)">
              {b.label}
            </text>
          );
        }
        return (
          <g key={b.label}>
            <rect
              className="morph"
              x={b.x}
              y={Y(b.p)}
              width={68}
              height={base - Y(b.p)}
              fill={b.pattern ? "url(#data-hatch)" : b.fill}
              stroke={b.pattern ? "var(--q-whirlpool)" : "none"}
            />
            <text x={b.x + 34} y={Y(b.p) - (b.label === "your guess" ? 24 : 8)} textAnchor="middle" fontSize={14} fontWeight={650} fill={b.label === "your guess" ? "var(--accent-strong)" : "var(--text)"}>
              {(b.p * 100).toFixed(0)}%
            </text>
            <text x={b.x + 34} y={base + 20} textAnchor="middle" fontSize={11} fill="var(--text-3)">
              {b.label}
            </text>
          </g>
        );
      })}
      <g {...drag.handleProps} data-dragging={drag.dragging ? "true" : undefined} className="guess-handle">
        <rect x={52} y={Y(guess) - 13} width={84} height={26} rx={13} fill="var(--accent-strong)" />
        <text x={94} y={Y(guess) + 5} textAnchor="middle" fontSize={12} fontWeight={650} fill="var(--on-accent)">
          drag me
        </text>
      </g>
      <text x={40} y={base + 40} fontSize={12} fill="var(--text-2)">
        Share of Row players choosing Top, $3.20 treatment
      </text>
    </svg>
  );
}

function LambdaChart({
  curves,
  lam,
  setLam,
  bestLam,
}: {
  curves: { grid: number[]; series: number[][] };
  lam: number;
  setLam: (v: number) => void;
  bestLam: number;
}) {
  const W = 560;
  const H = 300;
  const L = 46;
  const R = 130;
  const T = 20;
  const B = 245;
  const lo = Math.log(0.05);
  const hi = Math.log(60);
  const X = (l: number) => L + (W - R - L) * ((Math.log(Math.max(l, 0.05)) - lo) / (hi - lo));
  const Y = (p: number) => B - (B - T) * p;
  const colors = ["var(--text-3)", "var(--q-whirlpool)", "var(--q-landscape)"];
  const drag = useDragNumber({
    value: Math.log(lam),
    min: lo,
    max: hi,
    step: (hi - lo) / 200,
    onChange: (v) => setLam(Number(Math.exp(v).toFixed(3))),
    axis: "x",
    travelPx: 380,
    label: "Precision lambda",
    valueText: () => `lambda ${lam.toFixed(2)}`,
  });
  return (
    <>
    <svg viewBox={`0 0 ${W} ${H}`} className="demo-svg demo-svg-chart" role="img" aria-label={`Logit equilibrium probability of Top against precision lambda for the three treatments, with the observed frequencies marked. At lambda ${lam.toFixed(2)} the three predictions are ${curves.series.map((s) => (interp(curves.grid, s, lam) * 100).toFixed(0)).join(", ")} percent.`}>
      <line x1={L} y1={B} x2={W - R} y2={B} stroke="var(--text-3)" />
      <line x1={L} y1={T} x2={L} y2={B} stroke="var(--text-3)" />
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <text x={L - 8} y={Y(g) + 4} textAnchor="end" fontSize={11} fill="var(--text-3)" fontFamily="var(--mono)">
            {g * 100}%
          </text>
          <line x1={L} y1={Y(g)} x2={W - R} y2={Y(g)} stroke={g === 0.5 ? "var(--text-3)" : "var(--border)"} strokeWidth={g === 0.5 ? 1.6 : 1} strokeDasharray={g === 0.5 ? "6 3" : "2 5"} />
        </g>
      ))}
      <text x={L} y={B + 34} fontSize={11} fill="var(--text-3)">
        λ = 0.05
      </text>
      <text x={W - R} y={B + 34} textAnchor="end" fontSize={11} fill="var(--text-3)">
        λ = 60 (effectively Nash)
      </text>
      {TREATMENTS.map((t, i) => (
        <g key={t.id}>
          <line x1={L} y1={Y(t.dataTop)} x2={W - R} y2={Y(t.dataTop)} stroke={colors[i]} strokeWidth={1} strokeDasharray="1 4" opacity={0.9} />
          <text x={W - R + 6} y={Y(t.dataTop) + 4} fontSize={11} fill={colors[i]}>
            {(t.dataTop * 100).toFixed(0)}% observed
          </text>
        </g>
      ))}
      {curves.series.map((s, i) => (
        <path
          key={i}
          d={s.map((p, k) => `${k ? "L" : "M"}${X(curves.grid[k]).toFixed(1)},${Y(p).toFixed(1)}`).join("")}
          fill="none"
          stroke={colors[i]}
          strokeWidth={2.4}
          strokeDasharray={i === 0 ? "9 5" : i === 2 ? "2 3" : undefined}
        />
      ))}
      <line x1={X(bestLam)} y1={T} x2={X(bestLam)} y2={B} stroke="var(--text-3)" strokeWidth={1} strokeDasharray="4 4" />
      <text x={X(bestLam) + 5} y={T + 12} fontSize={10} fill="var(--text-3)">
        best fit λ={bestLam.toFixed(2)}
      </text>
      <g {...drag.handleProps} data-dragging={drag.dragging ? "true" : undefined} className="lam-handle">
        <line className="morph" x1={X(lam)} y1={T} x2={X(lam)} y2={B} stroke="var(--accent-strong)" strokeWidth={2.5} />
        <rect className="morph" x={X(lam) - 30} y={B + 4} width={60} height={24} rx={12} fill="var(--accent-strong)" />
        <text className="morph" x={X(lam)} y={B + 20} textAnchor="middle" fontSize={12} fontWeight={650} fill="var(--on-accent)">
          λ {lam.toFixed(1)}
        </text>
      </g>
      {TREATMENTS.map((t, i) => (
        <circle key={t.id} className="morph" cx={X(lam)} cy={Y(interp(curves.grid, curves.series[i], lam))} r={5} fill={colors[i]} stroke="var(--surface)" strokeWidth={1.5} />
      ))}
      <text x={L + 8} y={Y(0.5) - 7} fontSize={11} fill="var(--text-3)">
        Nash: 50% in all three, at every λ
      </text>
    </svg>
      <p className="lattice-legend">
        <strong>Solid</strong> is the $3.20 treatment, <strong>long-dashed</strong> the basic game,{" "}
        <strong>short-dashed</strong> the $0.44 treatment. Each treatment&apos;s finely-dotted horizontal line is what
        the subjects actually did; the heavy dashed line across the middle is Nash, which is 50% in all three at every λ.
      </p>
    </>
  );
}

function interp(grid: number[], series: number[], x: number) {
  if (x <= grid[0]) return series[0];
  if (x >= grid[grid.length - 1]) return series[series.length - 1];
  let k = 0;
  while (k < grid.length - 2 && grid[k + 1] < x) k++;
  const t = (x - grid[k]) / (grid[k + 1] - grid[k]);
  return series[k] + t * (series[k + 1] - series[k]);
}

function TravellerBars() {
  const rows = [
    { label: "R = 180", nash: 180, data: 201 },
    { label: "R = 5", nash: 180, data: 280 },
  ];
  const scale = (v: number) => ((v - 180) / 120) * 100;
  return (
    <div className="td-bars">
      {rows.map((r) => (
        <div key={r.label} className="td-row">
          <span className="td-label">{r.label}</span>
          <div className="td-track">
            <div className="td-nash" style={{ left: `${scale(r.nash)}%` }} title="Nash equilibrium claim: 180" />
            <div className="td-data" style={{ width: `${Math.max(1.5, scale(r.data))}%` }} />
            <span className="td-value">avg {r.data}</span>
          </div>
        </div>
      ))}
      <p className="td-key">
        Claims run from 180 to 300. The <span className="td-key-nash">▏</span> mark is the Nash equilibrium — 180 in both
        rows — and the bar is the observed average claim.
      </p>
      <div className="demo-readouts">
        <Readout label="Nash, both treatments" value="180" />
        <Readout label="observed, R = 180" value="201" />
        <Readout label="observed, R = 5" value="280" tone="whirlpool" />
      </div>
    </div>
  );
}
