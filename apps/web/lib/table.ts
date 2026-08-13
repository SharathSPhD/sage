/* Reading somebody else's CSV.
 *
 * Small on purpose: a delimiter sniff, RFC-4180 quoting, and column
 * inference that is only ever a suggestion — every guess it makes is shown
 * as a control the visitor can change, because inference on somebody else's
 * file is a guess however good it is.
 */

export interface Table {
  columns: string[];
  rows: string[][];
}

export function parseTable(text: string): Table {
  const clean = text.replace(/^﻿/, "").replace(/\r\n?/g, "\n").trim();
  if (clean === "") return { columns: [], rows: [] };
  const delimiter = sniff(clean);
  const lines = splitRecords(clean);
  const parsed = lines.map((line) => splitFields(line, delimiter));
  const width = Math.max(...parsed.map((r) => r.length));
  const first = parsed[0] ?? [];
  const headed = first.some((c) => c.trim() !== "" && !Number.isFinite(Number(c)));
  const columns = headed
    ? first.map((c, i) => (c.trim() === "" ? `column ${i + 1}` : c.trim()))
    : Array.from({ length: width }, (_, i) => `column ${i + 1}`);
  const body = headed ? parsed.slice(1) : parsed;
  return {
    columns,
    rows: body.filter((r) => r.some((c) => c.trim() !== "")).map((r) => pad(r, columns.length)),
  };
}

function pad(row: string[], n: number): string[] {
  const out = row.slice(0, n);
  while (out.length < n) out.push("");
  return out;
}

function sniff(text: string): string {
  const head = text.split("\n").slice(0, 5).join("\n");
  const counts: Record<string, number> = { ",": 0, "\t": 0, ";": 0, "|": 0 };
  for (const ch of head) if (ch in counts) counts[ch] += 1;
  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
}

/** Newlines inside quotes do not end a record. */
function splitRecords(text: string): string[] {
  const out: string[] = [];
  let current = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '"') {
      quoted = !quoted;
      current += ch;
    } else if (ch === "\n" && !quoted) {
      out.push(current);
      current = "";
    } else {
      current += ch;
    }
  }
  if (current.trim() !== "") out.push(current);
  return out;
}

function splitFields(line: string, delimiter: string): string[] {
  const out: string[] = [];
  let current = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (quoted && line[i + 1] === '"') {
        current += '"';
        i++;
      } else quoted = !quoted;
    } else if (ch === delimiter && !quoted) {
      out.push(current);
      current = "";
    } else current += ch;
  }
  out.push(current);
  return out.map((c) => c.trim());
}

export function columnValues(table: Table, index: number): string[] {
  return table.rows.map((r) => r[index] ?? "");
}

export function numericColumn(table: Table, index: number): number[] {
  return columnValues(table, index)
    .map((v) => Number(v.replace(/[$,%\s]/g, "")))
    .filter((v) => Number.isFinite(v));
}

export interface ColumnProfile {
  name: string;
  index: number;
  numericShare: number;
  distinct: number;
  looksLikeLevel: boolean;
  looksLikeCount: boolean;
  looksLikeLabel: boolean;
}

export function profile(table: Table): ColumnProfile[] {
  return table.columns.map((name, index) => {
    const raw = columnValues(table, index);
    const nums = raw.map((v) => Number(v.replace(/[$,%\s]/g, "")));
    const numericShare = raw.length ? nums.filter((v) => Number.isFinite(v)).length / raw.length : 0;
    const distinct = new Set(raw).size;
    const lower = name.toLowerCase();
    const wholeNumbers = nums.every((v) => !Number.isFinite(v) || Number.isInteger(v));
    return {
      name,
      index,
      numericShare,
      distinct,
      looksLikeLevel:
        numericShare > 0.9 && /price|bid|offer|level|rate|amount|cost|value|\$/.test(lower) && distinct > 1,
      looksLikeCount: numericShare > 0.9 && wholeNumbers && /count|n|obs|weight|freq|volume|qty/.test(lower),
      looksLikeLabel: numericShare < 0.5 || /week|date|time|period|round|id|store|label|group|region/.test(lower),
    };
  });
}

/** A first guess at the mapping, shown as editable controls. */
export function inferMapping(table: Table): { level: number; rival: number; count: number; group: number } {
  const cols = profile(table);
  const levels = cols.filter((c) => c.looksLikeLevel);
  const numeric = cols.filter((c) => c.numericShare > 0.9 && c.distinct > 2);
  const level = levels[0]?.index ?? numeric[0]?.index ?? (cols.length > 1 ? 1 : 0);
  const rival = levels[1]?.index ?? (numeric.find((c) => c.index !== level)?.index ?? -1);
  const count = cols.find((c) => c.looksLikeCount && c.index !== level && c.index !== rival)?.index ?? -1;
  const group = cols.find((c) => c.looksLikeLabel && c.index !== level && c.index !== rival)?.index ?? -1;
  return { level, rival, count, group };
}

/**
 * Turn a column of observed values into at most `maxLevels` levels, and count
 * how often each was chosen. Values that already sit on a small ladder are
 * kept exactly; a continuous column is binned to the midpoints of equal-width
 * bins, and the fact that it was binned is reported.
 */
export function toLevels(
  values: number[],
  maxLevels = 8,
): { levels: number[]; counts: number[]; binned: boolean } {
  const distinct = [...new Set(values.map((v) => Number(v.toFixed(6))))].sort((a, b) => a - b);
  if (distinct.length <= maxLevels) {
    const counts = distinct.map((d) => values.filter((v) => Math.abs(v - d) < 1e-6).length);
    return { levels: distinct, counts, binned: false };
  }
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const width = (hi - lo) / maxLevels || 1;
  const levels: number[] = [];
  const counts = new Array(maxLevels).fill(0);
  for (let i = 0; i < maxLevels; i++) levels.push(Number((lo + width * (i + 0.5)).toFixed(4)));
  for (const v of values) {
    const bin = Math.min(maxLevels - 1, Math.max(0, Math.floor((v - lo) / width)));
    counts[bin] += 1;
  }
  return { levels, counts, binned: true };
}
