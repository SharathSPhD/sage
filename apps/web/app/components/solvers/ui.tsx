"use client";

/* The parts every solver page is built from: a numeric field you can type into
 * or drag, the answer block, distribution bars, a curve over the decision
 * grid, and the one-line model note that sits under the result the way a stats
 * package prints its model line.
 */

import { useId, useState } from "react";

export const money = (v: number, digits = 2) =>
  `$${v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;
export const money0 = (v: number) => `$${Math.round(v).toLocaleString()}`;
export const pct = (p: number, digits = 0) => `${(100 * p).toFixed(digits)}%`;
export const num = (v: number, digits = 2) => v.toFixed(digits);
/** A whole-number count with thousands separators — flows, trips, minutes. */
export const count = (v: number) => Math.round(v).toLocaleString();

/** 3 significant figures, switching to exponent where a decimal would be noise. */
export function sig(v: number): string {
  if (v === 0) return "0";
  const magnitude = Math.abs(v);
  if (magnitude >= 1e5 || magnitude < 1e-3) return v.toExponential(2);
  return Number(v.toPrecision(3)).toString();
}

export interface FieldProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  help?: string;
  format?: (v: number) => string;
  unit?: string;
}

/** One input: type an exact value, or drag. Both write the same number. */
export function Field({ label, value, onChange, min, max, step, help, format }: FieldProps) {
  const id = useId();
  const [draft, setDraft] = useState<string | null>(null);

  const commit = (text: string) => {
    setDraft(text);
    const parsed = Number(text);
    if (text.trim() !== "" && Number.isFinite(parsed)) onChange(parsed);
  };

  return (
    <div className="knob">
      <span className="knob-label">
        <label htmlFor={id}>{label}</label>
        {format && <output aria-hidden="true">{format(value)}</output>}
      </span>
      <div className="knob-row">
        <input
          id={id}
          className="knob-number"
          type="number"
          inputMode="decimal"
          min={min}
          max={max}
          step={step}
          value={draft ?? String(value)}
          onChange={(e) => commit(e.target.value)}
          onBlur={() => setDraft(null)}
        />
        <input
          type="range"
          aria-label={label}
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => {
            setDraft(null);
            onChange(Number(e.target.value));
          }}
        />
      </div>
      {help && <span className="knob-help">{help}</span>}
    </div>
  );
}

/** The result. Headline first, then the figures that qualify it. */
export function Answer({
  headline,
  busy,
  error,
  children,
}: {
  headline: React.ReactNode;
  busy?: boolean;
  error?: string | null;
  children?: React.ReactNode;
}) {
  return (
    <section className="answer" aria-labelledby="answer-heading" aria-busy={busy ? "true" : "false"}>
      <div className="answer-head">
        <h2 id="answer-heading">Result</h2>
        <span className="solve-state" data-busy={busy ? "true" : "false"} role="status">
          {busy ? "solving" : ""}
        </span>
      </div>
      {error ? (
        <p className="studio-error" role="alert">
          {error}
        </p>
      ) : (
        <>
          <p className="answer-verb">{headline}</p>
          {children && <div className="answer-figures">{children}</div>}
        </>
      )}
    </section>
  );
}

export function Figure({
  label,
  value,
  note,
  tone,
}: {
  label: string;
  value: string;
  note?: string;
  tone?: "neutral" | "warn";
}) {
  return (
    <div>
      <div className="panel-label">{label}</div>
      <div className="reading" data-tone={tone}>
        {value}
      </div>
      {note && <p className="figure-note">{note}</p>}
    </div>
  );
}

/** Distribution over labelled levels. */
export function Bars({
  rows,
  limit = 10,
}: {
  rows: { label: string; p: number }[];
  limit?: number;
}) {
  const max = Math.max(...rows.map((r) => r.p), 1e-9);
  const shown = rows.length > limit ? [...rows].sort((a, b) => b.p - a.p).slice(0, limit) : rows;
  return (
    <>
      <ul className="rival-bars">
        {shown.map((r) => (
          <li key={r.label}>
            <span className="rb-label">{r.label}</span>
            <span className="rb-track">
              <span className="rb-fill" style={{ width: `${Math.max(0.6, (100 * r.p) / max)}%` }} />
            </span>
            <span className="rb-val">{pct(r.p, r.p < 0.1 ? 1 : 0)}</span>
          </li>
        ))}
      </ul>
      {rows.length > limit && (
        <p className="figure-note">{rows.length - limit} smaller levels not shown.</p>
      )}
    </>
  );
}

/** A curve over the decision grid, with the recommended level marked. */
export function Curve({
  x,
  y,
  markIndex,
  formatX,
  formatY,
  xLabel,
  yLabel,
}: {
  x: number[];
  y: number[];
  markIndex: number;
  formatX: (v: number) => string;
  formatY: (v: number) => string;
  xLabel: string;
  yLabel: string;
}) {
  const W = 560;
  const H = 190;
  const PAD = { l: 62, r: 14, t: 12, b: 34 };
  if (x.length < 2 || y.length < 2) return null;
  const yMin = Math.min(...y, 0);
  const yMax = Math.max(...y);
  const px = (i: number) => PAD.l + ((W - PAD.l - PAD.r) * i) / (x.length - 1);
  const py = (v: number) =>
    H - PAD.b - ((H - PAD.t - PAD.b) * (v - yMin)) / Math.max(yMax - yMin, 1e-12);
  const path = y.map((v, i) => `${i === 0 ? "M" : "L"}${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(" ");

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="curve-chart"
      role="img"
      aria-label={`${yLabel} against ${xLabel}. Highest at ${formatX(x[markIndex])}, ${formatY(y[markIndex])}. The table beside this chart carries the same numbers.`}
    >
      <line x1={PAD.l} y1={py(yMin)} x2={W - PAD.r} y2={py(yMin)} stroke="var(--border-bright)" />
      <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke="var(--border-bright)" />
      <text x={PAD.l - 8} y={py(yMax) + 4} textAnchor="end" fontSize="10" fill="var(--text-faint)" fontFamily="var(--mono)">
        {formatY(yMax)}
      </text>
      <text x={PAD.l - 8} y={py(yMin) + 4} textAnchor="end" fontSize="10" fill="var(--text-faint)" fontFamily="var(--mono)">
        {formatY(yMin)}
      </text>
      <path d={path} fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinejoin="round" />
      {markIndex >= 0 && markIndex < x.length && (
        <>
          <line
            x1={px(markIndex)}
            y1={PAD.t}
            x2={px(markIndex)}
            y2={H - PAD.b}
            stroke="var(--amber)"
            strokeDasharray="4 3"
          />
          <circle cx={px(markIndex)} cy={py(y[markIndex])} r="4.5" fill="var(--amber)" />
          <text
            x={Math.min(px(markIndex) + 7, W - PAD.r - 60)}
            y={PAD.t + 11}
            fontSize="10.5"
            fill="var(--amber)"
            fontFamily="var(--mono)"
          >
            {formatX(x[markIndex])}
          </text>
        </>
      )}
      <text x={PAD.l} y={H - 8} fontSize="10" fill="var(--text-faint)" fontFamily="var(--mono)">
        {formatX(x[0])}
      </text>
      <text x={W - PAD.r} y={H - 8} textAnchor="end" fontSize="10" fill="var(--text-faint)" fontFamily="var(--mono)">
        {formatX(x[x.length - 1])}
      </text>
      <text x={(W + PAD.l) / 2} y={H - 8} textAnchor="middle" fontSize="10" fill="var(--text-faint)">
        {xLabel}
      </text>
    </svg>
  );
}

/** The model line: what was solved, in the notation a stats package would print. */
export function ModelLine({ children }: { children: React.ReactNode }) {
  return <p className="model-line">{children}</p>;
}

export function Controls({ children, title = "Inputs" }: { children: React.ReactNode; title?: string }) {
  return (
    <section className="card controls" aria-label={title}>
      <h3>{title}</h3>
      <div className="knob-grid">{children}</div>
    </section>
  );
}

/** Comparative statics: one input swept, the recommendation at each value. */
export function Sweep({
  inputLabel,
  choices,
  chosen,
  onChoose,
  rows,
  outputLabel,
  valueLabel,
  note,
}: {
  inputLabel: string;
  choices: { key: string; label: string }[];
  chosen: string;
  onChoose: (key: string) => void;
  rows: { x: string; y: string; value: string; changed: boolean }[];
  outputLabel: string;
  valueLabel: string;
  note?: string;
}) {
  const id = useId();
  return (
    <section className="card" aria-labelledby={`${id}-h`}>
      <div className="sweep-head">
        <h3 id={`${id}-h`}>What moves it</h3>
        <label className="sweep-pick">
          <span className="visually-hidden">Input to sweep</span>
          <select value={chosen} onChange={(e) => onChoose(e.target.value)}>
            {choices.map((c) => (
              <option key={c.key} value={c.key}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <table className="alt-table">
        <thead>
          <tr>
            <th scope="col">{inputLabel}</th>
            <th scope="col">{outputLabel}</th>
            <th scope="col">{valueLabel}</th>
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={3}>Solving…</td>
            </tr>
          ) : (
            rows.map((r, i) => (
              <tr key={i} data-changed={r.changed}>
                <th scope="row">{r.x}</th>
                <td>{r.y}</td>
                <td>{r.value}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
      <p className="figure-note">
        {note ??
          "Every row is a separate solve with only this input changed. Rows that return a different recommendation are marked."}
      </p>
    </section>
  );
}
