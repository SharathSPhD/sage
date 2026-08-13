/* A miniature of what each problem type actually looks like once solved.
 *
 * These are illustrations, not readings — the shapes are fixed, so no number
 * is implied. They exist so the catalogue reads as a set of instruments
 * rather than a list of links.
 */

const W = 300;
const H = 84;

export function ProblemThumb({ id }: { id: string }) {
  const common = { viewBox: `0 0 ${W} ${H}`, className: "gallery-thumb", role: "presentation" as const };

  if (id === "pricing") {
    const pts = [8, 22, 40, 58, 70, 74, 68, 52, 34, 18];
    const d = pts
      .map((v, i) => `${i === 0 ? "M" : "L"}${12 + (i * (W - 24)) / (pts.length - 1)},${H - 10 - v * 0.82}`)
      .join(" ");
    return (
      <svg {...common} aria-hidden>
        <path d={`${d} L${W - 12},${H - 10} L12,${H - 10} Z`} fill="var(--series-1)" opacity="0.12" />
        <path d={d} fill="none" stroke="var(--series-1)" strokeWidth="2" strokeLinejoin="round" />
        <line x1={12 + (5 * (W - 24)) / 9} y1="8" x2={12 + (5 * (W - 24)) / 9} y2={H - 10} stroke="var(--border-strong)" />
        <circle cx={12 + (5 * (W - 24)) / 9} cy={H - 10 - 74 * 0.82} r="5.5" fill="var(--surface)" />
        <circle cx={12 + (5 * (W - 24)) / 9} cy={H - 10 - 74 * 0.82} r="3.5" fill="var(--series-1)" />
      </svg>
    );
  }

  if (id === "auction") {
    const bars = [4, 9, 17, 28, 41, 52, 44, 30, 17, 8, 3];
    return (
      <svg {...common} aria-hidden>
        {bars.map((v, i) => (
          <rect
            key={i}
            x={14 + i * ((W - 28) / bars.length)}
            y={H - 12 - v}
            width={(W - 28) / bars.length - 3}
            height={v}
            rx="3"
            fill={i === 5 ? "var(--series-1)" : "var(--series-2)"}
            opacity={i === 5 ? 1 : 0.55}
          />
        ))}
        <line x1="12" y1={H - 12} x2={W - 12} y2={H - 12} stroke="var(--border-strong)" />
      </svg>
    );
  }

  if (id === "electricity") {
    return (
      <svg {...common} aria-hidden>
        <rect x="16" y={H - 12 - 26} width="86" height="26" rx="3" fill="var(--series-3)" opacity="0.9" />
        <rect x="106" y={H - 12 - 46} width="78" height="46" rx="3" fill="var(--series-1)" opacity="0.9" />
        <rect x="188" y={H - 12 - 62} width="70" height="62" rx="3" fill="var(--series-3)" opacity="0.5" />
        <line x1="150" y1="8" x2="150" y2={H - 12} stroke="var(--text-2)" strokeWidth="1.5" />
        <line x1="12" y1={H - 12 - 46} x2={W - 12} y2={H - 12 - 46} stroke="var(--text-2)" strokeWidth="1.5" strokeDasharray="6 4" />
        <circle cx="150" cy={H - 12 - 46} r="5.5" fill="var(--surface)" />
        <circle cx="150" cy={H - 12 - 46} r="3.5" fill="var(--text)" />
      </svg>
    );
  }

  if (id === "routing") {
    const nodes: [number, number][] = [
      [40, 20], [110, 14], [180, 24], [250, 18],
      [30, 52], [98, 46], [168, 58], [244, 50],
      [64, 74], [136, 70], [206, 76], [270, 70],
    ];
    const edges: [number, number, number][] = [
      [0, 1, 3.4], [1, 2, 1.4], [2, 3, 2.2], [0, 4, 1.2], [1, 5, 4.4],
      [2, 6, 2.6], [3, 7, 1.3], [4, 5, 2.1], [5, 6, 3.6], [6, 7, 1.5],
      [4, 8, 1.1], [5, 9, 2.8], [6, 10, 1.7], [7, 11, 1.2], [8, 9, 2.4], [9, 10, 3.1], [10, 11, 1.4],
    ];
    return (
      <svg {...common} aria-hidden>
        {edges.map(([a, b, w], i) => (
          <line
            key={i}
            x1={nodes[a][0]}
            y1={nodes[a][1]}
            x2={nodes[b][0]}
            y2={nodes[b][1]}
            stroke={w > 3.2 ? "var(--danger)" : w > 2.3 ? "var(--warn)" : "var(--series-1)"}
            strokeWidth={w}
            strokeLinecap="round"
            opacity="0.85"
          />
        ))}
        {nodes.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r="3.6" fill="var(--surface)" stroke="var(--border-strong)" />
        ))}
      </svg>
    );
  }

  if (id === "allocation") {
    const grid = [
      [0, 1, 2, 3, 2],
      [1, 3, 5, 4, 2],
      [2, 5, 6, 3, 1],
      [3, 4, 3, 1, 0],
      [2, 2, 1, 0, 0],
    ];
    const ramp = ["var(--seq-1)", "var(--seq-2)", "var(--seq-3)", "var(--seq-4)", "var(--seq-5)", "var(--seq-6)", "var(--seq-7)"];
    const cell = 13;
    return (
      <svg {...common} aria-hidden>
        {grid.map((row, r) =>
          row.map((v, c) => (
            <rect
              key={`${r}-${c}`}
              x={W / 2 - (5 * (cell + 2)) / 2 + c * (cell + 2)}
              y={9 + r * (cell + 2)}
              width={cell}
              height={cell}
              rx="2.5"
              fill={ramp[v]}
            />
          )),
        )}
      </svg>
    );
  }

  // standards / payoff table
  const cells = [
    [6, 1, 2],
    [1, 5, 1],
    [2, 1, 3],
  ];
  const ramp = ["var(--seq-1)", "var(--seq-2)", "var(--seq-3)", "var(--seq-4)", "var(--seq-5)", "var(--seq-6)", "var(--seq-7)"];
  return (
    <svg {...common} aria-hidden>
      {cells.map((row, r) =>
        row.map((v, c) => (
          <rect
            key={`${r}-${c}`}
            x={W / 2 - (3 * 30) / 2 + c * 30}
            y={9 + r * 22}
            width={27}
            height={19}
            rx="3"
            fill={ramp[v]}
          />
        )),
      )}
    </svg>
  );
}
