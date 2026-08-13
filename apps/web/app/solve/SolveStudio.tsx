"use client";

/*
 * The solution surface.
 *
 * Every number on screen is computed from the numbers in the controls, in the
 * browser, on every change. A moment later the same game goes to the deployed
 * solver and the badge says whether the two agree. Nothing is cached, nothing
 * is pre-baked, and nothing appears that the inputs do not support.
 */

import Link from "next/link";
import { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { findFlips, lamFor, sharpnessFlip, solveSituation, type Flip, type Solution } from "../../lib/solve";
import { SITUATIONS, getSituation, type BuiltSituation, type Situation } from "../../lib/situations";
import { Term } from "../components/Term";

type Check = "pending" | "agree" | "differ" | "offline";

const SHARP_MIN = 0.6;
const SHARP_MAX = 120;

function pct(p: number): string {
  return `${(100 * p).toFixed(p >= 0.1 ? 0 : 1)}%`;
}

/** How to say, in one line, what the rival is doing. */
function rivalSentence(sol: Solution): string {
  const top = [...sol.rival].sort((a, b) => b.p - a.p);
  if (sol.rivalConcentration > 0.75) {
    return `They land on ${top[0].label} in ${pct(top[0].p)} of rounds. You can plan against one number.`;
  }
  if (sol.rivalConcentration < 0.25) {
    return "They are spread across every option. Nothing you do is safe against all of them — the value below already accounts for that.";
  }
  return `${top[0].label} is their most likely move at ${pct(top[0].p)}, but ${top[1].label} at ${pct(top[1].p)} is live. Plan for both.`;
}

function FlipLine({ f }: { f: Flip }) {
  return (
    <li>
      <span className="flip-knob">{f.knob}</span> <span className="flip-dir">{f.direction}</span>{" "}
      <span className="flip-at">{f.at}</span>
      <span className="flip-then"> → {f.becomes}</span>
    </li>
  );
}

export function SolveStudio({ fixedSituation }: { fixedSituation?: string }) {
  const [situationId, setSituationId] = useState(fixedSituation ?? SITUATIONS[0].id);
  const situation: Situation = useMemo(() => getSituation(situationId), [situationId]);

  // Keyed by situation. The alternative — one bag of numbers reset by an
  // effect — feeds the previous situation's values into the next one's payoff
  // builder for exactly one render, which is one NaN too many.
  const [byId, setById] = useState<Record<string, { values: Record<string, number>; sharpness: number }>>({});
  const state = byId[situationId] ?? { values: situation.defaults, sharpness: situation.defaultSharpness };
  const { values, sharpness } = state;
  const setValues = (v: Record<string, number>) => setById({ ...byId, [situationId]: { ...state, values: v } });
  const setSharpness = (n: number) => setById({ ...byId, [situationId]: { ...state, sharpness: n } });
  const resetNumbers = () => setById({ ...byId, [situationId]: { values: situation.defaults, sharpness: situation.defaultSharpness } });

  const [check, setCheck] = useState<Check>("pending");
  const [checkNote, setCheckNote] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  const built: BuiltSituation = useMemo(() => situation.build(values), [situation, values]);
  const solution = useMemo(() => solveSituation(built, sharpness), [built, sharpness]);

  // The sweeps below are a few hundred solves. Deferring them keeps the drag
  // at 60fps and lets the sensitivity text catch up a frame later.
  const lagged = useDeferredValue(state);
  const sensitivity = useMemo(
    () => findFlips(situation, lagged.values, lagged.sharpness),
    [situation, lagged],
  );
  const sharpFlip = useMemo(
    () => sharpnessFlip(situation, lagged.values, lagged.sharpness, SHARP_MIN, SHARP_MAX),
    [situation, lagged],
  );

  // The committed answer. Debounced, aborted on the next change, and compared
  // against what is already on screen rather than replacing it.
  useEffect(() => {
    setCheck("pending");
    const t = setTimeout(() => {
      abort.current?.abort();
      const ctl = new AbortController();
      abort.current = ctl;
      fetch("/api/v1/solve/qre", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ payoffs: [built.u1, built.u2], lam: lamFor(built, sharpness) }),
        signal: ctl.signal,
      })
        .then(async (r) => {
          if (!r.ok) throw new Error(String(r.status));
          return r.json();
        })
        .then((d: { sigma: number[][] }) => {
          if (ctl.signal.aborted) return;
          const gap = Math.max(...d.sigma[1].map((p: number, j: number) => Math.abs(p - solution.rival[j].p)));
          setCheck(gap < 1e-6 ? "agree" : "differ");
          setCheckNote(gap < 1e-6 ? null : `largest difference ${gap.toExponential(1)}`);
        })
        .catch((e) => {
          if ((e as Error).name === "AbortError") return;
          setCheck("offline");
        });
    }, 400);
    return () => clearTimeout(t);
  }, [built, sharpness, solution]);

  const rec = solution.recommended;
  const alternatives = solution.options
    .filter((o) => o.index !== rec.index)
    .sort((a, b) => b.value - a.value)
    .slice(0, 3);

  const badge =
    check === "agree"
      ? { tone: "ok" as const, text: "checked against the deployed solver" }
      : check === "differ"
        ? { tone: "warn" as const, text: `solver disagrees · ${checkNote}` }
        : check === "offline"
          ? { tone: undefined, text: "solved in this browser · solver unreachable" }
          : { tone: undefined, text: "solved in this browser · checking…" };

  return (
    <div className="studio">
      {!fixedSituation && (
        <div className="situation-picker" role="group" aria-label="Choose a situation">
          {SITUATIONS.map((s) => (
            <button
              key={s.id}
              type="button"
              data-on={s.id === situationId}
              aria-pressed={s.id === situationId}
              onClick={() => setSituationId(s.id)}
            >
              {s.name}
            </button>
          ))}
        </div>
      )}

      <p className="studio-decision">{situation.decision}</p>

      <section className="answer" aria-labelledby="answer-heading">
        <div className="answer-head">
          <h2 id="answer-heading">Do this</h2>
          <span className="badge" data-tone={badge.tone}>
            {badge.text}
          </span>
        </div>
        <p className="answer-verb">{built.decide(built.yourMoves[rec.index])}.</p>
        <div className="answer-figures">
          <div>
            <div className="panel-label">Worth</div>
            <div className="reading">{built.formatValue(rec.value)}</div>
            <p className="figure-note">{built.valueUnit}</p>
          </div>
          <div>
            <div className="panel-label">Most rounds it lands between</div>
            <div className="reading" data-tone="neutral">
              {built.formatValue(rec.lo)} – {built.formatValue(rec.hi)}
            </div>
            <p className="figure-note">8 times in 10. The spread is what the rival does, not measurement error.</p>
          </div>
          <div>
            <div className="panel-label">Worst single outcome</div>
            <div className="reading" data-tone="warn">
              {built.formatValue(rec.worst)}
            </div>
            <p className="figure-note">if they go to {rec.worstAgainst}</p>
          </div>
        </div>
      </section>

      <div className="answer-cols">
        <section className="card" aria-labelledby="rival-heading">
          <h3 id="rival-heading">What they are likely to do</h3>
          <p className="lead-note">{rivalSentence(solution)}</p>
          <ul className="rival-bars">
            {solution.rival.map((r) => (
              <li key={r.label}>
                <span className="rb-label">{r.label}</span>
                <span className="rb-track">
                  <span className="rb-fill" style={{ width: `${Math.max(0.6, 100 * r.p)}%` }} />
                </span>
                <span className="rb-val">{pct(r.p)}</span>
              </li>
            ))}
          </ul>
        </section>

        <section className="card" aria-labelledby="alt-heading">
          <h3 id="alt-heading">The alternatives, and what each costs you</h3>
          <table className="alt-table">
            <thead>
              <tr>
                <th scope="col">Instead</th>
                <th scope="col">Worth</th>
                <th scope="col">Gives up</th>
                <th scope="col">Worst case</th>
              </tr>
            </thead>
            <tbody>
              {alternatives.map((o) => (
                <tr key={o.index}>
                  <th scope="row">{o.label}</th>
                  <td>{built.formatValue(o.value)}</td>
                  <td className="alt-cost">−{built.formatValue(o.cost)}</td>
                  <td>{built.formatValue(o.worst)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {solution.nashOption && (
            <p className="lead-note">
              Assuming they play perfectly would put you on {solution.nashOption.label}, which gives up{" "}
              {built.formatValue(solution.nashOption.cost)} against a rival who is only human.
            </p>
          )}
        </section>
      </div>

      <div className="answer-cols">
        <section className="card" aria-labelledby="holds-heading">
          <h3 id="holds-heading">What has to be true</h3>
          <ul className="holds">
            <li>
              They are choosing too — reacting to you, not sitting still. If they are on a fixed schedule, this is the
              wrong model and <Link href="/diagnose">the fit check</Link> will say so.
            </li>
            <li>The options on the table are the ones listed. A move outside the ladder is outside the answer.</li>
            {sharpFlip.flipsAt === null ? (
              <li>
                Nothing about how sharply they chase their own best option changes the answer — it holds from
                near-random to near-perfect play.
              </li>
            ) : (
              <li>
                They chase their best option about this hard. Take the control below{" "}
                {sharpFlip.direction} {sharpFlip.flipsAt.toFixed(1)} and the answer becomes {sharpFlip.becomes}.
              </li>
            )}
            {sensitivity.holds.length > 0 && (
              <li>
                The answer does not depend on {sensitivity.holds.join(", ").toLowerCase()} — anywhere in the range on
                screen.
              </li>
            )}
          </ul>
        </section>

        <section className="card" aria-labelledby="flips-heading">
          <h3 id="flips-heading">What would change it</h3>
          {sensitivity.flips.length === 0 ? (
            <p className="lead-note">
              Nothing on screen flips it. Every control can go to either end of its range and the recommendation holds.
            </p>
          ) : (
            <ul className="flips">
              {sensitivity.flips.map((f) => (
                <FlipLine key={f.knob} f={f} />
              ))}
            </ul>
          )}
        </section>
      </div>

      <section className="card controls" aria-labelledby="controls-heading">
        <h3 id="controls-heading">Your numbers</h3>
        <p className="lead-note">Change any of these and the answer above moves as you drag.</p>
        <div className="knob-grid">
          {situation.knobs.map((k) => (
            <label key={k.key} className="knob">
              <span className="knob-label">
                {k.label}
                <output aria-hidden="true">{k.format(values[k.key])}</output>
              </span>
              <input
                type="range"
                aria-label={k.label}
                min={k.min}
                max={k.max}
                step={k.step}
                value={values[k.key]}
                onChange={(e) => setValues({ ...values, [k.key]: Number(e.target.value) })}
              />
              {k.help && <span className="knob-help">{k.help}</span>}
            </label>
          ))}
          <label className="knob">
            <span className="knob-label">
              {situation.sharpnessLabel}
              <output aria-hidden="true">{sharpness.toFixed(1)}</output>
            </span>
            <input
              type="range"
              aria-label={situation.sharpnessLabel}
              min={SHARP_MIN}
              max={SHARP_MAX}
              step={0.2}
              value={sharpness}
              onChange={(e) => setSharpness(Number(e.target.value))}
            />
            <span className="knob-help">
              Low means they wander; high means they find their best option every time.{" "}
              <Term
                term="Written λ"
                explain="λ, the logit precision: how strongly a choice responds to a difference in payoff. λ = 0 is a coin flip; λ → ∞ is perfect optimisation."
              />
            </span>
          </label>
        </div>
        <div className="controls-actions">
          <button
            type="button"
            onClick={resetNumbers}
          >
            Reset the numbers
          </button>
          <Link href={situation.href} className="control-link">
            Open the full {situation.name.toLowerCase()} scenario →
          </Link>
        </div>
      </section>

      <p className="source-note" data-illustrative={situation.illustrative}>
        <strong>{situation.illustrative ? "Illustrative numbers." : "Measured numbers."}</strong> {situation.sourceNote}
      </p>

      <p className="escape-hatch">
        Not sure this is even the right model for your market?{" "}
        <Link href="/diagnose">Check the fit against your own data</Link> before you trust the number above.
      </p>
    </div>
  );
}
