"use client";

/* Five decision rules, one simulated rival, one seed, up to 120 rounds.
 *
 * Each rule plays its own copy of the situation against its own copy of the
 * rival, driven from one seeded stream. The environment can be changed
 * mid-run; the change lands on every rule at the same round and is marked on
 * the chart.
 */

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { POLICIES, playRound, rng, type ArenaConfig, type PolicyId, type RoundState } from "../../lib/backtest";
import { SITUATIONS, getSituation } from "../../lib/situations";

const W = 720;
const H = 300;
const PAD = { l: 58, r: 12, t: 14, b: 30 };
const ROUNDS = 120;

interface Marker {
  round: number;
  label: string;
}

export function Arena() {
  const [situationId, setSituationId] = useState("procurement");
  const situation = useMemo(() => getSituation(situationId), [situationId]);

  // Keyed by situation: one bag of numbers reset by an effect would feed the
  // wrong situation's values into a payoff builder for one render.
  const [byId, setById] = useState<Record<string, Record<string, number>>>({});
  const values = byId[situationId] ?? situation.defaults;
  const setValues = (v: Record<string, number>) => setById((b) => ({ ...b, [situationId]: v }));
  const [cfg, setCfg] = useState<ArenaConfig>({
    rivalSharpness: 8,
    rivalHedge: 0.35,
    rivalNoise: 0.1,
    costPlusMarkup: 0.25,
  });
  const [history, setHistory] = useState<RoundState[]>([]);
  const [markers, setMarkers] = useState<Marker[]>([]);
  const [running, setRunning] = useState(false);
  const [seed, setSeed] = useState(20260813);
  const drawRef = useRef<() => number>(rng(seed));

  const built = useMemo(() => situation.build(values), [situation, values]);

  const reset = useCallback(() => {
    setRunning(false);
    setHistory([]);
    setMarkers([]);
    drawRef.current = rng(seed);
  }, [seed]);

  useEffect(() => {
    reset();
  }, [situationId, reset]);

  useEffect(() => {
    if (history.length >= ROUNDS) setRunning(false);
  }, [history.length]);

  const step = useCallback(() => {
    setHistory((h) => {
      const prev = h.length ? h[h.length - 1] : null;
      return [...h, playRound(built, cfg, cfg.rivalSharpness, prev, drawRef.current)];
    });
  }, [built, cfg]);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      setHistory((h) => {
        if (h.length >= ROUNDS) return h;
        const prev = h.length ? h[h.length - 1] : null;
        return [...h, playRound(built, cfg, cfg.rivalSharpness, prev, drawRef.current)];
      });
    }, 110);
    return () => clearInterval(id);
  }, [running, built, cfg]);

  const shock = (label: string, apply: () => void) => {
    apply();
    setMarkers((m) => [...m, { round: history.length, label }]);
  };

  const last = history.length ? history[history.length - 1] : null;
  const standings = useMemo(() => {
    if (!last) return [];
    return POLICIES.map((p) => ({
      ...p,
      total: last.cumulative[p.id],
      perRound: last.cumulative[p.id] / last.round,
    })).sort((a, b) => b.total - a.total);
  }, [last]);

  const leader = standings[0];
  const us = standings.find((s) => s.id === "sage");

  // Chart geometry.
  const yMax = last ? Math.max(...POLICIES.map((p) => last.cumulative[p.id]), 1) : 1;
  const yMin = last ? Math.min(...POLICIES.map((p) => last.cumulative[p.id]), 0) : 0;
  const xOf = (r: number) => PAD.l + ((W - PAD.l - PAD.r) * r) / Math.max(history.length, 20);
  const yOf = (v: number) => H - PAD.b - ((H - PAD.t - PAD.b) * (v - yMin)) / Math.max(yMax - yMin, 1e-9);

  const paths = POLICIES.map((p) => ({
    id: p.id,
    colour: p.colour,
    dash: p.dash,
    short: p.short,
    endY: last ? yOf(last.cumulative[p.id]) : 0,
    d: history
      .map((s, i) => `${i === 0 ? "M" : "L"}${xOf(i + 1).toFixed(1)},${yOf(s.cumulative[p.id]).toFixed(1)}`)
      .join(" "),
  }));

  const ties = (() => {
    if (!last) return [];
    const groups = new Map<string, string[]>();
    for (const p of POLICIES) {
      const key = last.cumulative[p.id].toFixed(6);
      groups.set(key, [...(groups.get(key) ?? []), p.name]);
    }
    return [...groups.values()].filter((g) => g.length > 1);
  })();

  const verdict = (() => {
    if (!last || last.round < 8 || !us || !leader) return null;
    if (leader.id === "sage") {
      const second = standings[1];
      return `After ${last.round} rounds the solver leads ${second.name.toLowerCase()} by ${built.formatValue(us.total - second.total)}.`;
    }
    return `After ${last.round} rounds ${leader.name.toLowerCase()} leads the solver by ${built.formatValue(leader.total - us.total)}.`;
  })();

  const note = (() => {
    if (!leader || leader.id === "sage") return null;
    if (leader.id === "bestResponseLast") {
      return "Best reply to last wins when the rival has stopped varying: it lands on their one move exactly, and hedging against a collapsed distribution buys insurance that is not needed. Raise the rival's randomness to see it reverse.";
    }
    if (leader.id === "costPlus") {
      return "Cost-plus leads when your payoff barely depends on the rival. On these numbers the strategic term is small, and the cost side is where the money is.";
    }
    if (leader.id === "nash") {
      return "Always-Nash leads when the rival is close to exact. Lower their precision and the gap closes, then reverses.";
    }
    if (leader.id === "match") {
      return "Matching leads when their move is the one worth making. It cannot lead the market and cannot be badly wrong.";
    }
    return null;
  })();

  return (
    <div className="studio">
      <div className="situation-picker" role="group" aria-label="Setting">
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

      <section className="card" aria-labelledby="arena-chart-heading">
        <div className="answer-head">
          <h2 id="arena-chart-heading">Cumulative value, round by round</h2>
          <span className="badge">
            {history.length} of {ROUNDS} rounds
          </span>
        </div>

        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="arena-chart"
          role="img"
          aria-label={`Cumulative value by decision rule over ${history.length} rounds. ${leader ? `${leader.name} leads.` : "Nothing has been run yet."} The standings table below carries the same numbers.`}
        >
          <line x1={PAD.l} y1={yOf(0)} x2={W - PAD.r} y2={yOf(0)} stroke="var(--border-bright)" strokeWidth="1" />
          <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke="var(--border-bright)" strokeWidth="1" />
          <text x={PAD.l - 8} y={yOf(yMax) + 4} textAnchor="end" fontSize="10" fill="var(--text-faint)" fontFamily="var(--mono)">
            {built.formatValue(yMax)}
          </text>
          <text x={PAD.l - 8} y={yOf(0) + 4} textAnchor="end" fontSize="10" fill="var(--text-faint)" fontFamily="var(--mono)">
            0
          </text>
          {markers.map((m, i) => (
            <g key={i}>
              <line x1={xOf(m.round)} y1={PAD.t} x2={xOf(m.round)} y2={H - PAD.b} stroke="var(--text-faint)" strokeDasharray="3 3" />
              <text x={xOf(m.round) + 4} y={PAD.t + 10} fontSize="9.5" fill="var(--text-dim)">
                {m.label}
              </text>
            </g>
          ))}
          {paths.map((p) => (
            <path
              key={p.id}
              d={p.d}
              fill="none"
              stroke={p.colour}
              strokeDasharray={p.dash === "none" ? undefined : p.dash}
              strokeWidth={p.id === "sage" ? 2.6 : 1.6}
              strokeLinejoin="round"
            />
          ))}
          {/* WCAG 1.4.1 — the dash pattern above and this label make each line
              identifiable without relying on its colour. Where two lines finish
              close together the labels are pushed apart and a leader line keeps
              each one attached to its own line, rather than stacking them into
              an unreadable clump. */}
          {last &&
            (() => {
              const MIN_GAP = 12;
              const placed = [...paths]
                .map((p) => ({ p, y: p.endY }))
                .sort((a, b) => a.y - b.y);
              for (let i = 1; i < placed.length; i++) {
                if (placed[i].y - placed[i - 1].y < MIN_GAP) placed[i].y = placed[i - 1].y + MIN_GAP;
              }
              const x = xOf(history.length);
              return placed.map(({ p, y }) => (
                <g key={`${p.id}-tag`}>
                  {Math.abs(y - p.endY) > 1.5 && (
                    <path
                      d={`M${x + 2},${p.endY} L${x + 8},${y - 3}`}
                      stroke={p.colour}
                      strokeWidth="0.8"
                      fill="none"
                      opacity="0.7"
                    />
                  )}
                  <text x={x + 10} y={y + 3} fontSize="9" fontFamily="var(--mono)" fill={p.colour}>
                    {p.short}
                  </text>
                </g>
              ));
            })()}
          <text x={W / 2} y={H - 8} textAnchor="middle" fontSize="10" fill="var(--text-faint)">
            rounds played
          </text>
        </svg>

        <div className="arena-actions">
          <button type="button" data-primary="true" onClick={() => setRunning((r) => !r)}>
            {running ? "Pause" : history.length ? "Keep going" : `Run ${ROUNDS} rounds`}
          </button>
          <button type="button" onClick={step} disabled={running}>
            One round
          </button>
          <button type="button" onClick={reset}>
            Start over
          </button>
          <label className="seed">
            Seed
            <input
              type="number"
              value={seed}
              onChange={(e) => {
                setSeed(Number(e.target.value));
              }}
              onBlur={reset}
            />
          </label>
        </div>

        <div className="arena-shocks">
          <span className="panel-label">Change the world mid-run</span>
          <button
            type="button"
            onClick={() => {
              const k = situation.knobs[0];
              shock("costs up", () =>
                setValues({ ...values, [k.key]: Math.min(k.max, values[k.key] + (k.max - k.min) * 0.18) }),
              );
            }}
          >
            Your costs jump
          </button>
          <button
            type="button"
            onClick={() =>
              shock("rival sharpens", () =>
                setCfg((c) => ({
                  ...c,
                  rivalSharpness: Math.min(120, c.rivalSharpness * 3),
                  rivalNoise: Math.max(0, c.rivalNoise - 0.08),
                })),
              )
            }
          >
            Rival gets disciplined
          </button>
          <button
            type="button"
            onClick={() =>
              shock("rival erratic", () =>
                setCfg((c) => ({
                  ...c,
                  rivalNoise: Math.min(0.9, c.rivalNoise + 0.3),
                  rivalHedge: Math.min(1, c.rivalHedge + 0.3),
                })),
              )
            }
          >
            Rival goes erratic
          </button>
        </div>
      </section>

      {verdict && (
        <section className="card verdict-card">
          <p className="answer-verb">{verdict}</p>
          {note && <p className="lead-note">{note}</p>}
          {ties.length > 0 && (
            <p className="lead-note">
              {ties.map((g) => `${g.join(" and ")} are level`).join("; ")} — they are choosing the same move on these
              settings, so their lines sit on top of each other.
            </p>
          )}
        </section>
      )}

      <div className="answer-cols">
        <section className="card" aria-labelledby="standings-heading">
          <h3 id="standings-heading">Standings</h3>
          {standings.length === 0 ? (
            <p className="lead-note">Press run. Every rule starts from the same round and the same rival.</p>
          ) : (
            <table className="alt-table">
              <thead>
                <tr>
                  <th scope="col">Rule</th>
                  <th scope="col">Total</th>
                  <th scope="col">Per round</th>
                </tr>
              </thead>
              <tbody>
                {standings.map((s) => (
                  <tr key={s.id} data-us={s.id === "sage"}>
                    <th scope="row">
                      <span className="swatch" style={{ background: s.colour }} aria-hidden="true" />
                      {s.name} <span className="short-tag">{s.short}</span>
                    </th>
                    <td>{built.formatValue(s.total)}</td>
                    <td>{built.formatValue(s.perRound)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        <section className="card" aria-labelledby="rules-heading">
          <h3 id="rules-heading">The rules</h3>
          <dl className="rules">
            {POLICIES.map((p) => (
              <div key={p.id}>
                <dt>
                  <span className="swatch" style={{ background: p.colour }} aria-hidden="true" />
                  {p.name} <span className="short-tag">{p.short}</span>
                </dt>
                <dd>{p.rule}</dd>
              </div>
            ))}
          </dl>
        </section>
      </div>

      <section className="card controls" aria-labelledby="env-heading">
        <h3 id="env-heading">The rival, and the world they are in</h3>
        <p className="lead-note">Change these at any point. The run continues from where it is.</p>
        <div className="knob-grid">
          <label className="knob">
            <span className="knob-label">
              Rival precision
              <output aria-hidden="true">{cfg.rivalSharpness.toFixed(1)}</output>
            </span>
            <input
              type="range"
              aria-label="Rival precision"
              min={0.6}
              max={120}
              step={0.2}
              value={cfg.rivalSharpness}
              onChange={(e) => setCfg({ ...cfg, rivalSharpness: Number(e.target.value) })}
            />
            <span className="knob-help">In units of the rival&apos;s own payoff spread.</span>
          </label>
          <label className="knob">
            <span className="knob-label">
              How much the rival hedges instead of betting you repeat
              <output aria-hidden="true">{(100 * cfg.rivalHedge).toFixed(0)}%</output>
            </span>
            <input
              type="range"
              aria-label="How much the rival hedges instead of betting you repeat"
              min={0}
              max={1}
              step={0.05}
              value={cfg.rivalHedge}
              onChange={(e) => setCfg({ ...cfg, rivalHedge: Number(e.target.value) })}
            />
          </label>
          <label className="knob">
            <span className="knob-label">
              Share of rounds the rival plays at random
              <output aria-hidden="true">{(100 * cfg.rivalNoise).toFixed(0)}%</output>
            </span>
            <input
              type="range"
              aria-label="Share of rounds the rival plays at random"
              min={0}
              max={0.9}
              step={0.05}
              value={cfg.rivalNoise}
              onChange={(e) => setCfg({ ...cfg, rivalNoise: Number(e.target.value) })}
            />
          </label>
          <label className="knob">
            <span className="knob-label">
              Cost-plus markup
              <output aria-hidden="true">{(100 * cfg.costPlusMarkup).toFixed(0)}%</output>
            </span>
            <input
              type="range"
              aria-label="Cost-plus markup"
              min={0.02}
              max={0.6}
              step={0.01}
              value={cfg.costPlusMarkup}
              onChange={(e) => setCfg({ ...cfg, costPlusMarkup: Number(e.target.value) })}
            />
          </label>
          {situation.knobs.slice(0, 2).map((k) => (
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
            </label>
          ))}
        </div>
      </section>

      <p className="model-line">
        {situation.model} The rival is a simulated player, not a record of anyone&apos;s behaviour, and a run is
        reproducible from its seed.
      </p>

      <p className="escape-hatch">
        For the single-round answer from the solver, <Link href="/solve">go to solve</Link>.
      </p>
    </div>
  );
}
