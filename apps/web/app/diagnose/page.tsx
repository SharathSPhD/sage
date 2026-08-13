"use client";

/**
 * /diagnose -- the practitioner front door.
 *
 * The whole route exists to move a visitor from "I have some data" to "I know what kind of
 * system this is, and what that changes" in under a minute. Five states, in order:
 *
 *   1  DROP       a file, or one of three bundled examples (one real, two generated —
 *                 each says which on the button itself)
 *   2  RECOGNISE  column-role inference + a report saying what this data can and cannot
 *                 identify (limits are reported as bounds, never as silence)
 *   3  THE PLANE  one dominant figure: the reading, with bands, on the reference cloud
 *   4  CONSEQUENCE what the quadrant changes, each line carrying the number that drove it
 *   5  TAKE IT    reproduce-in-python / export / why-should-I-believe-this
 *
 * Supersedes /tools, which did (2) and (3) with a textarea and had no (4) or (5).
 *
 * The browser only ever talks to /api/* on this origin; next.config.mjs proxies to the
 * backend server-side (same mechanism every other page here uses).
 */

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ResearchCrumb } from "../components/ResearchCrumb";

// ---------------------------------------------------------------------------------------
// Calibrated bands + the reference cloud. Same values as strataq.viz.REFERENCE_CLOUD.
// The four quadrant colours are CSS custom properties (--q-*) because the identical palette
// is used by strataq.viz and by the paper; -text variants are the AA-contrast versions for
// anything that is read rather than looked at.
// ---------------------------------------------------------------------------------------

const R_EDGE = 0.02;
const E_EDGE = 0.03;

const QUADRANTS = {
  landscape: {
    roman: "I",
    name: "Landscape",
    fill: "var(--q-landscape)",
    ink: "var(--q-landscape-text)",
    physics: "Reciprocal response, no persistent circulation — consistent with a potential game.",
    consequences: [
      "Comparative statics are trustworthy: a change you make moves the system where you expect.",
      "Pass-through is symmetric — you affect rivals about as much as they affect you.",
      "A static model of the other agents is adequate. Modelling their full distribution buys little.",
      "There is no cycle here, so effort spent timing one is wasted.",
    ],
    check: "If you think this is wrong, look for a cost shock that moved a rival much further than it moved you.",
  },
  "driven landscape": {
    roman: "II",
    name: "Driven landscape",
    fill: "var(--q-driven)",
    ink: "var(--q-driven-text)",
    physics: "Reciprocal response with persistent circulation — the system is being cycled from outside.",
    consequences: [
      "The circulation is exogenous — demand, schedules, cost — not the strategic interaction.",
      "Timing matters. Re-engineering the strategic structure does not.",
      "Comparative statics still hold within a phase of the cycle.",
      "Look for the driver before looking for a rival to blame.",
    ],
    check: "If you think this is wrong, remove the obvious periodicity and read it again: the result should collapse to the null.",
  },
  "stalled whirlpool": {
    roman: "III",
    name: "Stalled whirlpool",
    fill: "var(--q-stalled)",
    ink: "var(--q-stalled-text)",
    physics: "Asymmetric response with no persistent circulation — a structural leader, but nothing cycles.",
    consequences: [
      "One agent structurally leads. The asymmetry is the exploitable object.",
      "There is no cycle to time, so timing effort is wasted.",
      "Your pass-through and theirs are different sizes — measure both directions separately.",
      "A symmetric model of the interaction will misprice your response.",
    ],
    check: "If you think this is wrong, the asymmetry should vanish once you control for whatever makes the leader lead.",
  },
  whirlpool: {
    roman: "IV",
    name: "Whirlpool",
    fill: "var(--q-whirlpool)",
    ink: "var(--q-whirlpool-text)",
    physics: "Asymmetric response and persistent circulation — a non-equilibrium steady state.",
    consequences: [
      "Both structure and timing matter. This is the hardest regime.",
      "Optimising against a static model of the other agents is at its worst here.",
      "Model the rivals' distribution, not their point action.",
      "Expect cycles, and expect them to persist rather than damp out.",
    ],
    check: "If you think this is wrong, a long enough window should show the circulation dying rather than sustaining.",
  },
  undetermined: {
    roman: "—",
    name: "Not pinned down",
    fill: "var(--q-none)",
    ink: "var(--q-none-text)",
    physics: "One of the two coordinates is not identified by this data, so the quadrant is a set, not a point.",
    consequences: [
      "The set below is a true statement about your system — it is a bound, not a failure.",
      "To narrow it, supply the missing input. Section 2 names exactly which one.",
    ],
    check: "",
  },
} as const;

type QuadrantKey = keyof typeof QUADRANTS;

const REFERENCE = [
  { label: "Sioux Falls road network", r: 5.6e-17, epr: null, q: "landscape" },
  { label: "Dominick's retail panel", r: 0.00112, epr: null, q: "landscape" },
  { label: "Colonel Blotto", r: 0.118, epr: 0.098, q: "whirlpool" },
  { label: "rock–paper–scissors", r: 0.78, epr: 0.83, q: "whirlpool" },
  { label: "CAISO day-ahead", r: null, epr: 1.07, q: "driven landscape" },
] as const;

// ---------------------------------------------------------------------------------------

type Coord = { value: number | null; lo: number | null; hi: number | null; kind: string; note: string };
type Reading = {
  quadrant: QuadrantKey;
  live: QuadrantKey[];
  r: Coord;
  epr: Coord;
  limits: string[];
  warnings: string[];
  provenance: Record<string, string | number>;
};

function classify(r: Coord, e: Coord): { q: QuadrantKey; live: QuadrantKey[]; note: string | null } {
  const table: Record<string, QuadrantKey> = {
    "low|low": "landscape",
    "low|high": "driven landscape",
    "high|low": "stalled whirlpool",
    "high|high": "whirlpool",
  };
  let note: string | null = null;
  const sideR = (() => {
    if (r.kind === "absent") return "unknown";
    const lo = r.lo ?? r.value;
    const hi = r.hi ?? r.value;
    if (hi !== null && hi < R_EDGE) return "low";
    if (lo !== null && lo > R_EDGE) return "high";
    note = `The interval on R straddles the calibrated band edge (${R_EDGE}), so this data does not separate the reciprocal from the non-reciprocal half of the plane.`;
    return "unknown";
  })();
  const sideE = (() => {
    if (e.kind === "absent") return "unknown";
    if (e.kind === "lower_bound") return "high";
    if (e.kind === "upper_bound") return "low";
    return "unknown";
  })();
  if (sideR !== "unknown" && sideE !== "unknown") {
    const q = table[`${sideR}|${sideE}`];
    return { q, live: [q], note };
  }
  const live = (Object.entries(table) as [string, QuadrantKey][])
    .filter(([k]) => {
      const [a, b] = k.split("|");
      return (sideR === "unknown" || a === sideR) && (sideE === "unknown" || b === sideE);
    })
    .map(([, v]) => v);
  return { q: "undetermined", live, note };
}

// ---------------------------------------------------------------------------------------
// The plane. Hand-cut SVG, log axes, quadrant shading, bands for partial coordinates.
// role="img" + aria-labelledby: the whole figure is one image with one text alternative,
// and that alternative carries the actual reading, not just the chart type.
// ---------------------------------------------------------------------------------------

const R_LO = 3e-5,
  R_HI = 3,
  E_LO = 3e-4,
  E_HI = 3;
const W = 620,
  H = 400,
  PAD_L = 66,
  PAD_B = 48,
  PAD_T = 18,
  PAD_R = 18;

const sx = (r: number) => {
  const t = (Math.log10(Math.max(r, R_LO)) - Math.log10(R_LO)) / (Math.log10(R_HI) - Math.log10(R_LO));
  return PAD_L + t * (W - PAD_L - PAD_R);
};
const sy = (e: number) => {
  const t = (Math.log10(Math.max(e, E_LO)) - Math.log10(E_LO)) / (Math.log10(E_HI) - Math.log10(E_LO));
  return H - PAD_B - t * (H - PAD_B - PAD_T);
};

function fmtR(c: Coord): string {
  if (c.value === null) return "not identified";
  const pt = c.value < 0.01 ? c.value.toExponential(2) : c.value.toFixed(3);
  if (c.lo !== null && c.hi !== null) {
    return `${pt} (95% interval ${c.lo.toExponential(2)} to ${c.hi.toExponential(2)})`;
  }
  return `${pt} (point read, no interval)`;
}

function fmtE(c: Coord): string {
  if (c.value === null) return "not identified";
  if (c.kind === "lower_bound") return `at least ${(c.lo ?? c.value).toPrecision(3)} nats/step`;
  if (c.kind === "upper_bound") return `at most ${(c.hi ?? c.value).toPrecision(3)} nats/step`;
  return c.value.toPrecision(3);
}

/** The sentence a screen reader gets instead of the figure. Carries the reading itself. */
function planeAlt(reading: Reading | null): string {
  const axes =
    `Scatter plot of the irreversibility plane. The horizontal axis is response asymmetry R on a log ` +
    `scale from 1e-5 to 3, with the calibrated band edge at ${R_EDGE}. The vertical axis is dissipation ` +
    `EPR in nats per step on a log scale, with the band edge at ${E_EDGE}. The four quadrants are ` +
    `I landscape at bottom left, II driven landscape at top left, III stalled whirlpool at bottom right, ` +
    `IV whirlpool at top right.`;
  if (!reading) {
    return `${axes} Five reference systems are plotted for scale; they are listed in full beneath the figure. No reading is plotted yet.`;
  }
  const meta = QUADRANTS[reading.quadrant];
  const where =
    reading.quadrant === "undetermined"
      ? `Your reading is not pinned to one quadrant. It is consistent with ${reading.live
          .map((k) => `${QUADRANTS[k].roman} ${QUADRANTS[k].name.toLowerCase()}`)
          .join(", ")}.`
      : `Your reading sits in quadrant ${meta.roman}, ${meta.name.toLowerCase()}.`;
  return `${axes} Your reading: R is ${fmtR(reading.r)}; EPR is ${fmtE(reading.epr)}. ${where} Five reference systems are plotted for scale; they are listed in full beneath the figure.`;
}

function Plane({ reading, titleId, descId }: { reading: Reading | null; titleId: string; descId: string }) {
  const xEdge = sx(R_EDGE),
    yEdge = sy(E_EDGE);
  const q = (k: QuadrantKey) => QUADRANTS[k].fill;
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-labelledby={`${titleId} ${descId}`}
      className="plane"
    >
      <title id={titleId}>The irreversibility plane</title>
      <desc id={descId}>{planeAlt(reading)}</desc>

      <rect x={PAD_L} y={yEdge} width={xEdge - PAD_L} height={H - PAD_B - yEdge} fill={q("landscape")} opacity={0.16} />
      <rect x={PAD_L} y={PAD_T} width={xEdge - PAD_L} height={yEdge - PAD_T} fill={q("driven landscape")} opacity={0.16} />
      <rect x={xEdge} y={yEdge} width={W - PAD_R - xEdge} height={H - PAD_B - yEdge} fill={q("stalled whirlpool")} opacity={0.16} />
      <rect x={xEdge} y={PAD_T} width={W - PAD_R - xEdge} height={yEdge - PAD_T} fill={q("whirlpool")} opacity={0.16} />

      <line x1={xEdge} y1={PAD_T} x2={xEdge} y2={H - PAD_B} stroke="var(--text-faint)" strokeDasharray="4 3" strokeWidth={1} />
      <line x1={PAD_L} y1={yEdge} x2={W - PAD_R} y2={yEdge} stroke="var(--text-faint)" strokeDasharray="4 3" strokeWidth={1} />

      <text x={PAD_L + 8} y={yEdge - 8} fontSize={11} fontWeight={700} fill={QUADRANTS["landscape"].ink}>
        I landscape
      </text>
      <text x={PAD_L + 8} y={PAD_T + 14} fontSize={11} fontWeight={700} fill={QUADRANTS["driven landscape"].ink}>
        II driven landscape
      </text>
      <text x={W - PAD_R - 8} y={yEdge - 8} fontSize={11} fontWeight={700} fill={QUADRANTS["stalled whirlpool"].ink} textAnchor="end">
        III stalled whirlpool
      </text>
      <text x={W - PAD_R - 8} y={PAD_T + 14} fontSize={11} fontWeight={700} fill={QUADRANTS["whirlpool"].ink} textAnchor="end">
        IV whirlpool
      </text>

      {REFERENCE.map((ref) =>
        ref.r !== null && ref.epr !== null ? (
          <circle key={ref.label} cx={sx(ref.r)} cy={sy(ref.epr)} r={4} fill={QUADRANTS[ref.q].ink} opacity={0.75} />
        ) : ref.r !== null ? (
          <line
            key={ref.label}
            x1={sx(ref.r)}
            y1={PAD_T}
            x2={sx(ref.r)}
            y2={H - PAD_B}
            stroke={QUADRANTS[ref.q].ink}
            strokeDasharray="1 4"
            opacity={0.7}
          />
        ) : (
          <line
            key={ref.label}
            x1={PAD_L}
            y1={sy(ref.epr!)}
            x2={W - PAD_R}
            y2={sy(ref.epr!)}
            stroke={QUADRANTS[ref.q].ink}
            strokeDasharray="1 4"
            opacity={0.7}
          />
        )
      )}

      {reading &&
        (() => {
          const c = QUADRANTS[reading.quadrant].ink;
          const hasR = reading.r.value !== null;
          const hasE = reading.epr.value !== null;
          return (
            <g>
              {hasR && reading.r.lo !== null && reading.r.hi !== null && (
                <rect
                  x={sx(reading.r.lo)}
                  y={PAD_T}
                  width={Math.max(2, sx(reading.r.hi) - sx(reading.r.lo))}
                  height={H - PAD_B - PAD_T}
                  fill={c}
                  opacity={0.22}
                />
              )}
              {hasR && <line x1={sx(reading.r.value!)} y1={PAD_T} x2={sx(reading.r.value!)} y2={H - PAD_B} stroke={c} strokeWidth={2} />}
              {hasE && <line x1={PAD_L} y1={sy(reading.epr.value!)} x2={W - PAD_R} y2={sy(reading.epr.value!)} stroke={c} strokeWidth={2} />}
              {hasR && hasE && (
                <g>
                  <circle cx={sx(reading.r.value!)} cy={sy(reading.epr.value!)} r={9.5} fill="var(--bg)" />
                  <circle cx={sx(reading.r.value!)} cy={sy(reading.epr.value!)} r={6.5} fill={c} stroke="var(--text)" strokeWidth={1.2} />
                </g>
              )}
            </g>
          );
        })()}

      <line x1={PAD_L} y1={H - PAD_B} x2={W - PAD_R} y2={H - PAD_B} stroke="var(--border-bright)" strokeWidth={1} />
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={H - PAD_B} stroke="var(--border-bright)" strokeWidth={1} />
      {[-4, -3, -2, -1, 0].map((p) => (
        <g key={`x${p}`}>
          <line x1={sx(10 ** p)} y1={H - PAD_B} x2={sx(10 ** p)} y2={H - PAD_B + 4} stroke="var(--border-bright)" strokeWidth={1} />
          <text x={sx(10 ** p)} y={H - PAD_B + 17} fontSize={10} textAnchor="middle" fill="var(--text-dim)" className="mono">
            1e{p}
          </text>
        </g>
      ))}
      {[-3, -2, -1, 0].map((p) => (
        <g key={`y${p}`}>
          <line x1={PAD_L - 4} y1={sy(10 ** p)} x2={PAD_L} y2={sy(10 ** p)} stroke="var(--border-bright)" strokeWidth={1} />
          <text x={PAD_L - 8} y={sy(10 ** p) + 3} fontSize={10} textAnchor="end" fill="var(--text-dim)" className="mono">
            1e{p}
          </text>
        </g>
      ))}
      <text x={(PAD_L + W - PAD_R) / 2} y={H - 8} fontSize={11} textAnchor="middle" fill="var(--text-dim)">
        response asymmetry R (local)
      </text>
      <text
        x={15}
        y={(PAD_T + H - PAD_B) / 2}
        fontSize={11}
        textAnchor="middle"
        fill="var(--text-dim)"
        transform={`rotate(-90 15 ${(PAD_T + H - PAD_B) / 2})`}
      >
        dissipation EPR, nats/step (global)
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------------------------

type ReportLine = { kind: "note" | "bound" | "limit"; text: string };

/**
 * A numeric column, classified.
 *
 * `counter` is the important one. Nearly every CSV a practitioner has starts with a row
 * index, a week number or an evenly-spaced timestamp, and such a column has by far the
 * largest raw variance in the file — 1..840 has variance 58,800 against a price series'
 * 504. Choosing "the column that varies most" therefore chooses the counter every time,
 * and a counter is a perfectly monotone ramp, which is the single most irreversible series
 * that exists. It would come back "detected" for any file at all. So counters are detected
 * by their constant step and excluded, and what is left is ranked by *relative* spread,
 * which is scale-free — a price in dollars and the same price in cents rank identically.
 */
type NumericColumn = {
  index: number;
  name: string;
  values: number[];
  kind: "counter" | "constant" | "data";
  spread: number;
};

type Table = {
  columns: NumericColumn[];
  suggested: number | null; // position within `columns`, not the file column index
  rows: number;
  notes: ReportLine[];
};

const EPS = 1e-9;

function classifyColumn(index: number, name: string, values: number[]): NumericColumn {
  const n = values.length;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
  if (sd === 0) return { index, name, values, kind: "constant", spread: 0 };
  if (n >= 3) {
    const step = values[1] - values[0];
    const constantStep =
      step !== 0 &&
      values.every((v, i) => i === 0 || Math.abs(v - values[i - 1] - step) <= EPS * Math.max(1, Math.abs(step)));
    if (constantStep) return { index, name, values, kind: "counter", spread: sd };
  }
  // Coefficient of variation, with a fall-back to the raw sd for mean-zero data
  // (returns, residuals) where a CV is undefined rather than small.
  const spread = Math.abs(mean) > EPS ? sd / Math.abs(mean) : sd;
  return { index, name, values, kind: "data", spread };
}

function parseTable(text: string): Table {
  const notes: ReportLine[] = [];
  const rows = text.trim().split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (rows.length === 0) {
    return { columns: [], suggested: null, rows: 0, notes: [{ kind: "limit", text: "This file has no rows in it. Nothing to read yet." }] };
  }
  const split = (l: string) => l.split(/[,\t;]|\s+/).filter((s) => s.length > 0);
  const first = split(rows[0]);
  const headerLooksTextual = first.some((c) => Number.isNaN(Number(c)));
  const body = headerLooksTextual ? rows.slice(1) : rows;
  if (headerLooksTextual) notes.push({ kind: "note", text: `Header row: ${first.join(", ")}.` });
  if (body.length === 0) {
    return { columns: [], suggested: null, rows: 0, notes: [...notes, { kind: "limit", text: "This file has a header and no data rows." }] };
  }

  const width = Math.max(...body.slice(0, 50).map((l) => split(l).length));
  const columns: NumericColumn[] = [];
  for (let c = 0; c < width; c++) {
    const values = body.map((l) => Number(split(l)[c])).filter((x) => Number.isFinite(x));
    // A column has to be numeric in most rows to be a candidate at all; this is what
    // drops timestamp and label columns without having to guess at date formats.
    if (values.length < body.length * 0.8) continue;
    columns.push(classifyColumn(c, headerLooksTextual && first[c] ? first[c] : `column ${c + 1}`, values));
  }

  const usable = columns.filter((c) => c.kind === "data");
  let suggested: number | null = null;
  if (usable.length > 0) {
    const best = usable.reduce((a, b) => (b.spread > a.spread ? b : a));
    suggested = columns.indexOf(best);
  }

  const counters = columns.filter((c) => c.kind === "counter");
  if (counters.length > 0) {
    notes.push({
      kind: "note",
      text: `Ignoring ${counters.map((c) => `"${c.name}"`).join(", ")} — the values go up by a constant step, so ${
        counters.length > 1 ? "they are counters" : "it is a counter"
      }, not a measurement. A counter is a straight ramp and would read as irreversible whatever else is in the file.`,
    });
  }
  if (usable.length === 0) {
    notes.push({
      kind: "limit",
      text: "Every numeric column in this file is either a counter or constant. There is nothing here to test — add the column that holds the measurement.",
    });
  }
  return { columns, suggested, rows: body.length, notes };
}

/** What can and cannot be said about a chosen column, given only its length and kind. */
function describeSeries(col: NumericColumn, pickedByHand: boolean): ReportLine[] {
  const series = col.values;
  const name = col.name;
  const report: ReportLine[] = [];
  report.push({
    kind: "note",
    text: pickedByHand
      ? `Reading "${name}" — ${series.length} observations, taken in the order they appear.`
      : `Reading "${name}" — of the columns that are measurements rather than counters, it varies the most relative to its own size. ${series.length} observations, taken in the order they appear. Change it below if that is the wrong column.`,
  });
  if (col.kind === "counter") {
    report.push({
      kind: "bound",
      text: `"${name}" goes up by a constant step, which makes it a counter rather than a measurement. A counter is a straight ramp: it will come back irreversible, and that says nothing whatever about your system. Read this result as an artefact of the column choice.`,
    });
  }
  if (series.length > 20000) {
    report.push({
      kind: "limit",
      text: `This host reads at most 20,000 points at a time and this column has ${series.length}. Trim it, or run the same call locally with pip install strataq — the snippet in section 5 is the whole of it.`,
    });
  }
  if (series.length < 50) {
    report.push({
      kind: "limit",
      text: `Too short to test. The irreversibility test needs at least 50 observations in time order and this has ${series.length}. That is a limit of the sample, not a problem with the file.`,
    });
  } else if (series.length < 300) {
    report.push({
      kind: "bound",
      text: `At ${series.length} observations the test is underpowered — it reaches about 80% power from 300 onwards. If it finds nothing, read that as "this sample is too short to tell", not as "there is nothing there".`,
    });
  }
  return report;
}

/**
 * Two of these are generated and one is not, and the difference is stated on the button
 * itself rather than in a footnote. `caiso.csv` is the same committed artifact the /markets
 * page reads — real CAISO SP15 day-ahead prices — so a visitor's first reading is of a real
 * market. The other two are written to land in known places on the plane, and nothing read
 * off them is a finding about any real retailer.
 */
const EXAMPLES = [
  {
    key: "caiso",
    label: "CAISO day-ahead prices",
    provenance: "real" as const,
    detail: "840 hourly prices, SP15 hub, July 2026 — the series behind the /markets reading.",
  },
  {
    key: "random-walk",
    label: "Random walk (at the null)",
    provenance: "synthetic" as const,
    detail: "500 generated points with no cycle and no time direction. Should not escape the null.",
  },
  {
    key: "whirlpool",
    label: "Edgeworth cycle",
    provenance: "synthetic" as const,
    detail: "600 generated points: slow undercutting, then a jump back up. Strongly time-asymmetric.",
  },
] as const;

export default function DiagnosePage() {
  const [raw, setRaw] = useState("");
  const [sourceName, setSourceName] = useState<string | null>(null);
  const [chiText, setChiText] = useState("");
  const [chiSeText, setChiSeText] = useState("");
  const [reading, setReading] = useState<Reading | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [openDrawer, setOpenDrawer] = useState(false);
  const [copied, setCopied] = useState(false);
  const [dragging, setDragging] = useState(false);
  // null = "whatever the inference suggested"; a number overrides it.
  const [colOverride, setColOverride] = useState<number | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  // Static ids: the route renders this component exactly once, and static ids keep
  // the aria-labelledby / htmlFor wiring legible in the DOM inspector.
  const seriesId = "d-series";
  const chiId = "d-chi";
  const chiSeId = "d-chi-se";
  const colId = "d-column";
  const planeTitleId = "d-plane-title";
  const planeDescId = "d-plane-desc";

  const table = useMemo(() => (raw.trim() ? parseTable(raw) : null), [raw]);

  /** The chosen column and everything that follows from it: null when nothing is readable. */
  const parsed = useMemo(() => {
    if (!table) return null;
    const pos =
      colOverride !== null && colOverride >= 0 && colOverride < table.columns.length && table.columns[colOverride].kind !== "constant"
        ? colOverride
        : table.suggested;
    const col = pos === null ? null : table.columns[pos];
    const series = col ? col.values : [];
    const report = col
      ? [...table.notes, ...describeSeries(col, colOverride !== null && colOverride === pos)]
      : table.notes;
    return { series, report, column: col, position: pos };
  }, [table, colOverride]);
  const chiParsedOk = useMemo(() => {
    const rows = chiText.trim().split(/\r?\n/).filter((l) => l.trim());
    if (rows.length < 2) return false;
    const m = rows.map((l) => l.split(/[,\s;]+/).filter(Boolean).map(Number));
    return m.every((r) => r.length === m.length && r.every(Number.isFinite));
  }, [chiText]);

  useEffect(() => {
    if (!copied) return;
    const t = setTimeout(() => setCopied(false), 2200);
    return () => clearTimeout(t);
  }, [copied]);

  const parseMatrix = (t: string): number[][] | null => {
    const rows = t.trim().split(/\r?\n/).filter((l) => l.trim());
    if (rows.length < 2) return null;
    const m = rows.map((l) => l.split(/[,\s;]+/).filter(Boolean).map(Number));
    return m.every((r) => r.length === m.length && r.every(Number.isFinite)) ? m : null;
  };

  const takeFile = async (f: File) => {
    setSourceName(f.name);
    setReading(null);
    setErr(null);
    setColOverride(null);
    setRaw(await f.text());
  };

  const post = async (path: string, body: unknown) => {
    const r = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const detail = await r.json().catch(() => null);
      throw new Error(detail?.detail ?? `HTTP ${r.status}`);
    }
    return r.json();
  };

  const run = useCallback(async () => {
    setBusy(true);
    setErr(null);
    try {
      const limits: string[] = [];
      const warnings: string[] = [];
      const prov: Record<string, string | number> = {};

      let rCoord: Coord = { value: null, lo: null, hi: null, kind: "absent", note: "no response matrix supplied" };
      const chi = parseMatrix(chiText);
      if (chi) {
        const chiSe = parseMatrix(chiSeText);
        const res = await post("/api/v1/toolkit/reciprocity", { chi, chi_se: chiSe ?? null });
        rCoord = {
          value: res.r,
          lo: res.ci_low ?? null,
          hi: res.ci_high ?? null,
          kind: chiSe ? "interval" : "point",
          note: chiSe
            ? "Monte-Carlo interval from the standard errors you gave."
            : "Point read. With standard errors this becomes an interval.",
        };
        if (!chiSe) {
          limits.push(
            "R is a single number here, not an interval, because the response matrix came without standard errors. It will still be plotted, but which side of the band edge it falls on is not something this read can settle."
          );
        }
        (res.warnings ?? []).forEach((w: string) => warnings.push(w));
        prov["reciprocity"] = "strataq.toolkit.reciprocity_read";
      } else {
        limits.push(
          "R is not identified by this data. R comes from a perturbation experiment: shock one agent's costs or payoffs and observe how the others move. A time series on its own cannot supply it."
        );
      }

      let eCoord: Coord = { value: null, lo: null, hi: null, kind: "absent", note: "no trajectory supplied" };
      if (parsed && parsed.series.length >= 50 && parsed.series.length <= 20000) {
        const res = await post("/api/v1/toolkit/irreversibility", {
          series: parsed.series,
          n_bins: 3,
          n_surrogates: 200,
          alpha_level: 0.01,
          seed: 0,
        });
        const edge = res.null_quantile ?? res.null_median ?? res.statistic;
        eCoord = res.detected
          ? {
              value: res.statistic,
              lo: edge,
              hi: null,
              kind: "lower_bound",
              note: "This escapes its detailed-balance null. The certified statement is the escape, not the point value.",
            }
          : {
              value: res.statistic,
              lo: null,
              hi: edge,
              kind: "upper_bound",
              note: "No escape from the null. That puts a ceiling on EPR; it is not evidence that EPR is zero.",
            };
        (res.warnings ?? []).forEach((w: string) => warnings.push(w));
        prov["irreversibility"] = "strataq.toolkit.irreversibility_test";
        if (parsed.column) prov["column"] = parsed.column.name;
        prov["n_observations"] = parsed.series.length;
        prov["n_bins"] = 3;
        prov["n_surrogates"] = 200;
        prov["alpha_level"] = 0.01;
        prov["seed"] = 0;
        if (typeof res.p_value === "number") prov["p_value"] = res.p_value;
      } else {
        limits.push(
          "EPR is not identified by this data. EPR needs an observed sequence of joint states — a time series of at least 50 points, not a cross-section."
        );
      }

      const { q, live, note } = classify(rCoord, eCoord);
      if (note) warnings.push(note);
      setReading({ quadrant: q, live, r: rCoord, epr: eCoord, limits, warnings, provenance: prov });
      window.setTimeout(() => resultRef.current?.focus(), 0);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setErr(
        `The reading did not come back (${msg}). Nothing is wrong with your data — this is the backend. Try again; the status dot in the top bar shows whether it is up.`
      );
    } finally {
      setBusy(false);
    }
  }, [chiText, chiSeText, parsed]);

  const snippet = useMemo(() => {
    if (!reading) return "";
    return [
      "# pip install strataq",
      "import strataq",
      "",
      chiParsedOk
        ? `chi = ${JSON.stringify(parseMatrix(chiText))}`
        : "chi = None       # supply a perturbation experiment to identify R",
      parseMatrix(chiSeText)
        ? `chi_se = ${JSON.stringify(parseMatrix(chiSeText))}`
        : "chi_se = None    # standard errors turn R from a point into an interval",
      parsed?.series.length
        ? `series = [...]   # your ${parsed.series.length} observations, in time order`
        : "series = None    # supply a time series to identify EPR",
      "",
      "# same arguments this page used",
      "d = strataq.diagnose(chi=chi, chi_se=chi_se, series=series, n_surrogates=200, seed=0)",
      "print(d)            # the verdict",
      "print(d.explain())  # every band, null and limit",
      "d.plot()            # this point, on the reference cloud",
    ].join("\n");
  }, [reading, chiText, chiSeText, chiParsedOk, parsed]);

  const download = (name: string, body: string, mime = "text/plain") => {
    const url = URL.createObjectURL(new Blob([body], { type: mime }));
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  };

  const meta = reading ? QUADRANTS[reading.quadrant] : null;
  const canRun = !!table || chiParsedOk;

  return (
    <div className="wrap diagnose">
      <ResearchCrumb />
      <header>
        <p className="kicker">diagnose</p>
        <h1>What kind of system is this?</h1>
        <p className="lede">
          Give it a record of what the agents actually did. You get back a position in the
          irreversibility plane, a name for the regime, and a list of what that regime changes about
          how you should act. Where the data cannot settle something, the page says so and shows you
          the bound instead.
        </p>
      </header>

      <section aria-labelledby="d-s1" className="card">
        <h2 id="d-s1">
          <span className="step">1</span> Your data
        </h2>

        <button
          type="button"
          className="dropzone"
          data-dragging={dragging}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={async (e) => {
            e.preventDefault();
            setDragging(false);
            const f = e.dataTransfer.files?.[0];
            if (f) await takeFile(f);
          }}
          onClick={() => fileRef.current?.click()}
        >
          <strong>Choose a CSV file</strong>
          <span>or drag one here, paste the numbers below, or start from an example</span>
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".csv,.tsv,.txt"
          className="visually-hidden"
          tabIndex={-1}
          aria-hidden="true"
          onChange={async (e) => {
            const f = e.target.files?.[0];
            if (f) await takeFile(f);
          }}
        />

        <p className="hint" id="d-examples-hint">
          Start from one of these instead. The first is <strong>real</strong> — the committed CAISO
          artifact the <Link href="/markets">Markets</Link> reading is built on. The other two are{" "}
          <strong>generated</strong>, written to land in known parts of the plane so you can see the
          instrument agree with an answer that is known in advance; nothing read off them is a
          finding about any real market. Each button says which it is.
        </p>
        <div className="examples" role="group" aria-describedby="d-examples-hint" aria-label="Example series">
          {EXAMPLES.map((ex) => (
            <button
              key={ex.key}
              type="button"
              onClick={async () => {
                setErr(null);
                setReading(null);
                setColOverride(null);
                const t = await fetch(`/examples/${ex.key}.csv`)
                  .then((r) => r.text())
                  .catch(() => "");
                if (!t) {
                  setErr("That example did not load. Refresh the page and try again.");
                  return;
                }
                setSourceName(`${ex.key}.csv (${ex.provenance === "real" ? "real data" : "generated"})`);
                setRaw(t);
              }}
            >
              <span>{ex.label}</span>
              <span className="ex-detail">{ex.detail}</span>
              <span className="badge" data-provenance={ex.provenance}>
                {ex.provenance === "real" ? "real data" : "generated"}
              </span>
            </button>
          ))}
        </div>

        <label htmlFor={seriesId} className="panel-label">
          Or paste the series — one observation per row, in time order
        </label>
        <textarea
          id={seriesId}
          value={raw}
          onChange={(e) => {
            setRaw(e.target.value);
            setSourceName(null);
            setReading(null);
            setColOverride(null);
          }}
          rows={5}
          spellCheck={false}
          placeholder={"time,price\n1,1.70\n2,1.72"}
        />
        {/* The live region is always in the DOM — one inserted at announce time is
            missed by several screen readers. */}
        <p className="hint" aria-live="polite">
          {sourceName ? (
            <>
              Loaded <strong>{sourceName}</strong>.
            </>
          ) : (
            " "
          )}
        </p>

        <details>
          <summary>Add a response matrix, if you have run a perturbation experiment</summary>
          <p className="hint">
            Rows and columns are agents; entry (i, j) is how far agent i moved when agent j was
            shocked. Without it the response axis stays unidentified and the verdict comes back as a
            set of quadrants rather than one.
          </p>
          <label htmlFor={chiId} className="panel-label">
            Response matrix χ — one row per line
          </label>
          <textarea
            id={chiId}
            value={chiText}
            onChange={(e) => setChiText(e.target.value)}
            rows={3}
            spellCheck={false}
            placeholder={"1.07 0.0028\n0.0005 0.968"}
          />
          <label htmlFor={chiSeId} className="panel-label">
            Standard errors on χ — same shape (recommended: without them R has no interval)
          </label>
          <textarea
            id={chiSeId}
            value={chiSeText}
            onChange={(e) => setChiSeText(e.target.value)}
            rows={3}
            spellCheck={false}
            placeholder={"0.04 0.0011\n0.0004 0.031"}
          />
        </details>
      </section>

      {canRun && (
        <section aria-labelledby="d-s2" className="card">
          <h2 id="d-s2">
            <span className="step">2</span> What this data can identify
          </h2>
          {table && table.columns.length > 1 && (
            <div className="mapping">
              <label htmlFor={colId} className="panel-label">
                Column to read
              </label>
              <select
                id={colId}
                value={parsed?.position ?? ""}
                onChange={(e) => {
                  setColOverride(e.target.value === "" ? null : Number(e.target.value));
                  setReading(null);
                }}
              >
                {/* Nothing was inferable (every column is a counter or constant), so the
                    control must have a real option matching its own value rather than
                    silently displaying a column it is not reading. */}
                {parsed?.position === null && (
                  <option value="" disabled>
                    Nothing inferred — choose a column
                  </option>
                )}
                {table.columns.map((c, i) => (
                  <option key={c.index} value={i} disabled={c.kind === "constant"}>
                    {c.name}
                    {c.kind === "counter" ? " — counter" : c.kind === "constant" ? " — constant, nothing to read" : ""}
                    {i === table.suggested ? " (suggested)" : ""}
                  </option>
                ))}
              </select>
            </div>
          )}
          <ul className="report">
            {(parsed?.report ?? []).map((line, i) => (
              <li key={i} className={line.kind === "limit" ? "limit" : line.kind === "bound" ? "bound" : ""}>
                {line.kind !== "note" && <span className="tag">{line.kind === "limit" ? "Not identified" : "Bound"}</span>}
                {line.text}
              </li>
            ))}
            {!parsed?.column && (
              <li className="limit">
                <span className="tag">Not identified</span>
                No readable time series, so EPR stays open. Paste or drop one above and the vertical axis fills in.
              </li>
            )}
            {chiParsedOk ? (
              <li>
                A square response matrix was read, so R is identified.
                {!parseMatrix(chiSeText) && " Without standard errors it comes back as a point rather than an interval."}
              </li>
            ) : (
              <li className="limit">
                <span className="tag">Not identified</span>
                No response matrix, so R stays open. It cannot be recovered from a time series — it needs a
                perturbation experiment.
              </li>
            )}
          </ul>
          <button type="button" data-primary="true" disabled={busy || !canRun} onClick={run} aria-busy={busy}>
            {busy ? "Reading — 200 surrogate series…" : "Read this data"}
          </button>
          {err && (
            <p className="error" role="alert">
              {err}
            </p>
          )}
        </section>
      )}

      {reading && meta && (
        <>
          <section aria-labelledby="d-s3" className="card">
            <h2 id="d-s3">
              <span className="step">3</span> Where it sits
            </h2>
            <Plane reading={reading} titleId={planeTitleId} descId={planeDescId} />

            {/* Focus lands here when the read returns, so the verdict is the next thing a
                screen reader or keyboard user meets. No role="status" as well: the focus
                move already announces it, and a live region would say it twice. */}
            <div className="verdict" style={{ borderColor: meta.fill }} tabIndex={-1} ref={resultRef}>
              <span className="badge verdict-roman" style={{ color: meta.ink, borderColor: meta.fill }}>
                {meta.roman}
              </span>
              <div>
                <strong style={{ color: meta.ink }}>{meta.name}</strong>
                <p>{meta.physics}</p>
                <p className="coords mono">
                  R = {fmtR(reading.r)}
                  <br />
                  EPR = {fmtE(reading.epr)}
                </p>
                {reading.quadrant === "undetermined" && (
                  <p className="live">
                    Still consistent with:{" "}
                    {reading.live.map((k) => `${QUADRANTS[k].roman} ${QUADRANTS[k].name.toLowerCase()}`).join(" · ")}
                  </p>
                )}
              </div>
            </div>

            {reading.warnings.length > 0 && (
              <div className="warnings" style={{ marginTop: "1rem" }}>
                {reading.warnings.map((w) => (
                  <p className="w" key={w}>
                    {w}
                  </p>
                ))}
              </div>
            )}

            <details className="reference-key">
              <summary>What the other marks on the plane are</summary>
              <ul>
                {REFERENCE.map((ref) => (
                  <li key={ref.label}>
                    <strong>{ref.label}</strong> — {ref.r !== null ? `R = ${ref.r.toExponential(2)}` : "R not measured"};{" "}
                    {ref.epr !== null ? `EPR = ${ref.epr.toPrecision(3)}` : "EPR not measured"}; quadrant{" "}
                    {QUADRANTS[ref.q].roman} {QUADRANTS[ref.q].name.toLowerCase()}. Drawn as a line where only one
                    coordinate is known.
                  </li>
                ))}
              </ul>
            </details>
          </section>

          <section aria-labelledby="d-s4" className="card">
            <h2 id="d-s4">
              <span className="step">4</span> What that changes
            </h2>
            <ol className="consequences">
              {meta.consequences.map((c) => (
                <li key={c}>{c}</li>
              ))}
            </ol>
            {meta.check && (
              <p className="check">
                <strong>Check us:</strong> {meta.check}
              </p>
            )}
          </section>

          <section aria-labelledby="d-s5" className="card">
            <h2 id="d-s5">
              <span className="step">5</span> Take it with you
            </h2>
            <div className="actions">
              <button
                type="button"
                onClick={async () => {
                  try {
                    await navigator.clipboard.writeText(snippet);
                    setCopied(true);
                  } catch {
                    setErr("Copying is blocked in this browser. The snippet is below — select it and copy by hand.");
                  }
                }}
              >
                {copied ? "Copied" : "Copy the Python"}
              </button>
              <button type="button" onClick={() => download("diagnosis.json", JSON.stringify(reading, null, 2), "application/json")}>
                Download JSON
              </button>
              <button
                type="button"
                onClick={() =>
                  download(
                    "diagnosis.csv",
                    `coordinate,value,lo,hi,kind\nR,${reading.r.value},${reading.r.lo},${reading.r.hi},${reading.r.kind}\nEPR,${reading.epr.value},${reading.epr.lo},${reading.epr.hi},${reading.epr.kind}\n`,
                    "text/csv"
                  )
                }
              >
                Download CSV
              </button>
              <button type="button" onClick={() => setOpenDrawer((v) => !v)} aria-expanded={openDrawer} aria-controls="d-evidence">
                {openDrawer ? "Hide the evidence" : "Why should I believe this?"}
              </button>
            </div>
            <p className="hint" aria-live="polite">
              {copied ? "Snippet copied to the clipboard." : " "}
            </p>
            <pre className="snippet">{snippet}</pre>

            <div className="drawer" id="d-evidence" hidden={!openDrawer}>
              <h3>What this data could not settle</h3>
              {reading.limits.length ? (
                <ul>
                  {reading.limits.map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ul>
              ) : (
                <p>Both coordinates were identified. Nothing was left open.</p>
              )}

              <h3>Warnings raised during the read</h3>
              {reading.warnings.length ? (
                <ul>
                  {reading.warnings.map((w) => (
                    <li key={w}>{w}</li>
                  ))}
                </ul>
              ) : (
                <p>None.</p>
              )}

              <h3>Where the band edges come from</h3>
              <ul>
                <li>R band edges are {R_EDGE} and 0.30, calibrated on systems whose answer is known in advance.</li>
                <li>
                  The Sioux Falls road network — an exact potential game — reads R = 5.6e-17. Colonel Blotto reads 0.12;
                  rock–paper–scissors 0.69 to 0.87.
                </li>
                <li>The EPR threshold is escape from a detailed-balance null, never a fixed number.</li>
              </ul>

              <h3>Scope, and what these numbers are not</h3>
              <ul>
                <li>
                  R is zero if and only if the normalised game is potential, at every λ. The <em>magnitude</em> of R
                  scales with λ — never compare magnitudes read at different λ.
                </li>
                <li>R is a ratio of Frobenius norms and is unbounded above; values greater than 1 do occur.</li>
                <li>
                  The two coordinates share a zero and are otherwise independent observables. Neither is a proxy for the
                  other.
                </li>
              </ul>

              <h3>Provenance</h3>
              <ul>
                {Object.entries(reading.provenance).map(([k, v]) => (
                  <li key={k}>
                    <code>{k}</code>: {String(v)}
                  </li>
                ))}
              </ul>
              <p className="hint">
                Want to watch the instrument read zero where zero is provably right?{" "}
                <Link href="/situations/routing">Sioux Falls</Link> · <Link href="/situations/allocation">Colonel Blotto</Link>
              </p>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
