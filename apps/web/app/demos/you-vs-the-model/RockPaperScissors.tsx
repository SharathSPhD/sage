"use client";

import { useCallback, useMemo, useState } from "react";
import { RPS, fitLambda, softmax } from "../../../lib/demos/gametheory";
import { LAMBDA_MARKS } from "../../../lib/demos/landmarks";
import { Readout, Widget } from "../components/chrome";
import { useReducedMotion } from "../components/motion";

const NAMES = ["Rock", "Paper", "Scissors"];
const GLYPH = ["✊", "✋", "✌"];
const MODEL_LAMBDA = 3;

interface Round {
  you: number;
  model: number;
  /** The mix the model actually drew from, which is what the estimator conditions on. */
  mix: number[];
  result: number;
  predicted: number | null;
}

/** The model's own play: a logit response to its running count of your choices. */
function modelMix(history: number[]): number[] {
  const counts = [1, 1, 1];
  for (const h of history) counts[h] += 1;
  const total = counts[0] + counts[1] + counts[2];
  const belief = counts.map((c) => c / total);
  const eu = [0, 1, 2].map((j) => [0, 1, 2].reduce((acc, i) => acc + -RPS[i][j] * belief[i], 0));
  return softmax(eu, MODEL_LAMBDA);
}

export function RockPaperScissors() {
  const [rounds, setRounds] = useState<Round[]>([]);
  const reduced = useReducedMotion();

  const history = useMemo(() => rounds.map((r) => r.you), [rounds]);
  const nextMix = useMemo(() => modelMix(history), [history]);

  const fit = useMemo(() => {
    if (rounds.length < 5) return null;
    const payoffRows = rounds.map((r) => RPS.map((row) => row.reduce((acc, v, j) => acc + v * r.mix[j], 0)));
    return fitLambda(history, payoffRows);
  }, [rounds, history]);

  /** What the model expects you to do next, at YOUR fitted precision. */
  const prediction = useMemo(() => {
    if (!fit || fit.flat) return null;
    const eu = RPS.map((row) => row.reduce((acc, v, j) => acc + v * nextMix[j], 0));
    const p = softmax(eu, fit.lambda);
    const best = p.indexOf(Math.max(...p));
    return { p, best };
  }, [fit, nextMix]);

  const scored = rounds.filter((r) => r.predicted !== null);
  const hits = scored.filter((r) => r.predicted === r.you).length;
  const wins = rounds.filter((r) => r.result > 0).length;
  const losses = rounds.filter((r) => r.result < 0).length;

  const play = useCallback(
    (you: number) => {
      setRounds((prev) => {
        const hist = prev.map((r) => r.you);
        const mix = modelMix(hist);
        let draw = Math.random();
        let model = 2;
        for (let k = 0; k < 3; k++) {
          draw -= mix[k];
          if (draw < 0) {
            model = k;
            break;
          }
        }
        let predicted: number | null = null;
        if (prev.length >= 5) {
          const payoffRows = prev.map((r) => RPS.map((row) => row.reduce((a, v, j) => a + v * r.mix[j], 0)));
          const f = fitLambda(hist, payoffRows);
          if (f && !f.flat) {
            const eu = RPS.map((row) => row.reduce((a, v, j) => a + v * mix[j], 0));
            const p = softmax(eu, f.lambda);
            predicted = p.indexOf(Math.max(...p));
          }
        }
        return [...prev, { you, model, mix, result: RPS[you][model], predicted }];
      });
    },
    [],
  );

  const last = rounds[rounds.length - 1];
  const trail = useMemo(() => {
    const out: number[] = [];
    for (let n = Math.max(5, rounds.length - 5); n < rounds.length; n++) {
      const sub = rounds.slice(0, n);
      if (sub.length < 5) continue;
      const payoffRows = sub.map((r) => RPS.map((row) => row.reduce((a, v, j) => a + v * r.mix[j], 0)));
      const f = fitLambda(
        sub.map((r) => r.you),
        payoffRows,
      );
      if (f) out.push(f.lambda);
    }
    return out;
  }, [rounds]);

  return (
    <>
      <Widget
        hook="Play. That is the whole instruction."
        lede={
          <p>
            The model opposite you is a logit responder: it counts what you have played and answers that count with a
            distribution, not a best guess. Play a few rounds however you like — deliberately, randomly, stubbornly.
          </p>
        }
        consequence={
          <>
            Nothing here is scripted. The model draws its move from its own distribution before it sees yours, and every
            number on this page is computed from the rounds you just played.
          </>
        }
        maths={
          <>
            <p>
              The model plays <code>softmax(λ · u₂(belief), λ = 3)</code> against a Laplace-smoothed count of your
              choices. Its move is a draw from that distribution, not its mode, so it is beatable and it does not lock
              into a pattern.
            </p>
            <p>
              Payoffs are the standard rock–paper–scissors matrix: +1 for a win, −1 for a loss, 0 for a draw. The
              implementation is <code>apps/web/lib/demos/gametheory.ts</code>.
            </p>
          </>
        }
      >
        <div className="rps-play">
          <div className="rps-buttons">
            {NAMES.map((n, i) => (
              <button
                key={n}
                type="button"
                className="btn rps-btn"
                onClick={() => play(i)}
                data-expected={prediction?.best === i ? "true" : undefined}
              >
                <span className="rps-glyph" aria-hidden>
                  {GLYPH[i]}
                </span>
                {n}
              </button>
            ))}
          </div>
          <p className="rps-last" aria-live="polite">
            {last
              ? `Round ${rounds.length}: you ${NAMES[last.you]}, model ${NAMES[last.model]} — ${
                  last.result > 0 ? "you win" : last.result < 0 ? "the model wins" : "a draw"
                }.`
              : "No rounds played yet."}
          </p>
          <div className="demo-readouts">
            <Readout label="you – model" value={`${wins} – ${losses}`} />
            <Readout label="rounds" value={String(rounds.length)} />
            <Readout
              label="your λ̂"
              value={fit ? (fit.flat ? "not yet" : fit.lambda.toFixed(2)) : "—"}
              hand
              live
            />
          </div>
          {rounds.length > 0 ? (
            <button type="button" className="btn demo-reset" onClick={() => setRounds([])}>
              Start again
            </button>
          ) : null}
        </div>
      </Widget>

      <Widget
        hook="Where your play sits"
        lede={
          <p>
            Your rationality parameter λ̂ is fitted by maximum likelihood on the choices you actually made, against the
            payoffs you actually faced. It is the same estimator the project runs on market data.
          </p>
        }
        consequence={
          fit && !fit.flat ? (
            <>
              λ̂ = {fit.lambda.toFixed(2)}, 95% profile interval [{fit.ciLow.toFixed(2)}, {fit.ciHigh.toFixed(2)}]. A
              wide interval is not a failure of the fit; it is the honest statement that {rounds.length} choices cannot
              pin one number down.
            </>
          ) : (
            <>
              Play at least five rounds. Below that, and whenever no precision on the grid fits better than any
              other, the estimator refuses to quote a number instead of quoting a meaningless one.
            </>
          )
        }
        maths={
          <>
            <p>
              λ̂ = argmax<sub>λ</sub> Σ<sub>t</sub> log softmax(λ · u₁(mix<sub>t</sub>))[a<sub>t</sub>] over a grid of 401
              points spanning λ ∈ [0, 20], where mix<sub>t</sub> is the distribution the model actually drew from in
              round t. The interval is the profile-likelihood set within 1.92 log-likelihood of the peak, which is the
              χ²(1) 95% cut.
            </p>
            <p>
              The flat-likelihood guard is deliberate and matches the library's behaviour:{" "}
              <code>benchmarks/results/toolkit_verdicts.json</code> records{" "}
              <code>flat_likelihood_warned = 1</code> as an acceptance case — the instrument warns instead of quoting.
            </p>
            <p>
              Marks on the scale: λ = 1.2 is the bench every reciprocity and dynamics benchmark in this repository is
              read at; λ ≈ 4.78 is a least-squares logit fit to Goeree &amp; Holt&apos;s (2001) matching-pennies
              subjects, computed on the{" "}
              <a href="/demos/ten-little-treasures">Ten Little Treasures</a> page from their published frequencies.
            </p>
          </>
        }
      >
        <LambdaAxis lambda={fit && !fit.flat ? fit.lambda : null} trail={reduced ? [] : trail} ci={fit && !fit.flat ? [fit.ciLow, fit.ciHigh] : null} />
      </Widget>

      <Widget
        hook="Now it plays your λ̂ back at you"
        lede={
          <p>
            Once your precision is fitted, the model can run the same logit rule forward with{" "}
            <em>your</em> number in it and say what you are about to do — before you do it. The expected move is ringed
            on the buttons above.
          </p>
        }
        consequence={
          scored.length > 2 ? (
            <>
              It has called {hits} of {scored.length} ({((hits / scored.length) * 100).toFixed(0)}%). Chance is 33%. A
              score near chance means you are playing close to uniform — which is the correct thing to do in this game,
              and is exactly what a low λ̂ says about you.
            </>
          ) : (
            <>Play past five rounds and the prediction starts scoring itself.</>
          )
        }
        maths={
          <>
            <p>
              The prediction is softmax(λ̂ · u₁(mix<sub>next</sub>)), evaluated before your click and stored with the
              round. The scoreboard compares its mode with what you actually chose; no round is re-scored after the fact.
            </p>
            <p>
              This is the honest version of &ldquo;the model knows you&rdquo;. In rock–paper–scissors the equilibrium is
              uniform, so a well-played opponent is unpredictable and a high hit rate is evidence you drifted off
              equilibrium, not that the model is clever.
            </p>
          </>
        }
      >
        <div className="rps-predict">
          {prediction ? (
            <p aria-live="polite">
              It expects <strong>{NAMES[prediction.best]}</strong> next ({(prediction.p[prediction.best] * 100).toFixed(0)}
              % of its probability), and it committed to that before you clicked.
            </p>
          ) : (
            <p>Not enough rounds to predict yet.</p>
          )}
          <div className="demo-readouts">
            <Readout label="called right" value={scored.length ? `${hits} / ${scored.length}` : "—"} live />
            <Readout label="chance" value="33%" />
          </div>
        </div>
      </Widget>
    </>
  );
}

function LambdaAxis({
  lambda,
  trail,
  ci,
}: {
  lambda: number | null;
  trail: number[];
  ci: [number, number] | null;
}) {
  const W = 520;
  const H = 160;
  const pad = 34;
  const X = (v: number) => pad + (W - 2 * pad) * (Math.log10(1 + Math.min(v, 20)) / Math.log10(21));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="demo-svg demo-svg-axis" role="img" aria-label="Fitted precision on a scale from a coin flip to effectively Nash, with measured reference points marked">
      <line x1={pad} y1={96} x2={W - pad} y2={96} stroke="var(--text-3)" strokeWidth={2} />
      {LAMBDA_MARKS.map((m) => (
        <g key={m.label}>
          <line x1={X(m.lambda)} y1={88} x2={X(m.lambda)} y2={104} stroke="var(--text-3)" strokeWidth={1.5} />
          <text x={X(m.lambda)} y={122} textAnchor="middle" fontSize={11} fill="var(--text-3)">
            {m.label}
          </text>
          <text x={X(m.lambda)} y={136} textAnchor="middle" fontSize={10} fill="var(--text-3)" fontFamily="var(--mono)">
            λ={m.lambda}
          </text>
        </g>
      ))}
      {ci ? (
        <rect
          x={X(ci[0])}
          y={90}
          width={Math.max(2, X(ci[1]) - X(ci[0]))}
          height={12}
          fill="var(--accent)"
          opacity={0.18}
        />
      ) : null}
      {(lambda === null ? [] : trail).map((t, i) => (
        <circle key={i} cx={X(t)} cy={96} r={6} fill="var(--accent)" opacity={0.25 * ((i + 1) / (trail.length + 1))} />
      ))}
      {lambda === null ? (
        <text x={pad} y={40} fontSize={13} fill="var(--text-3)">
          Play five rounds to place yourself.
        </text>
      ) : (
        <>
          <circle className="morph" cx={X(lambda)} cy={96} r={9} fill="var(--accent-strong)" stroke="var(--surface)" strokeWidth={2} />
          <text className="morph" x={X(lambda)} y={70} textAnchor="middle" fontSize={16} fontWeight={700} fill="var(--accent-strong)">
            λ̂ = {lambda.toFixed(2)}
          </text>
          <text className="morph" x={X(lambda)} y={52} textAnchor="middle" fontSize={11} fill="var(--text-2)">
            you
          </text>
        </>
      )}
    </svg>
  );
}
