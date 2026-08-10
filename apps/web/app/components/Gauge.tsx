"use client";

/* A 240° analog arc gauge. value is clamped to [min, max]; danger marks the
   threshold fraction of the arc after which the needle turns amber → red. */
export function Gauge({
  value,
  min = 0,
  max = 1,
  label,
  format,
  danger,
}: {
  value: number | undefined;
  min?: number;
  max?: number;
  label: string;
  format: (v: number) => string;
  danger?: number; // value at which the reading is "hot"
}) {
  const frac =
    value === undefined ? 0 : Math.min(1, Math.max(0, (value - min) / (max - min)));
  const hot = value !== undefined && danger !== undefined && value >= danger;
  const start = -210; // degrees
  const sweep = 240;
  const angle = start + sweep * frac;
  const polar = (deg: number, r: number) => {
    const rad = (deg * Math.PI) / 180;
    return [50 + r * Math.cos(rad), 50 + r * Math.sin(rad)];
  };
  const [ax, ay] = polar(start, 38);
  const [bx, by] = polar(start + sweep, 38);
  const [nx, ny] = polar(angle, 30);
  const dangerFrac =
    danger === undefined ? null : Math.min(1, Math.max(0, (danger - min) / (max - min)));
  const dangerTick = dangerFrac === null ? null : polar(start + sweep * dangerFrac, 38);

  return (
    <div style={{ textAlign: "center" }}>
      <svg viewBox="0 0 100 78" style={{ width: "100%", maxWidth: 170 }} aria-label={label}>
        <path
          d={`M ${ax} ${ay} A 38 38 0 1 1 ${bx} ${by}`}
          fill="none"
          stroke="var(--border-bright)"
          strokeWidth="5"
          strokeLinecap="round"
        />
        <path
          d={`M ${ax} ${ay} A 38 38 0 ${frac > 0.5 ? 1 : 0} 1 ${polar(angle, 38)[0]} ${polar(angle, 38)[1]}`}
          fill="none"
          stroke={hot ? "var(--red)" : "var(--accent)"}
          strokeWidth="5"
          strokeLinecap="round"
          style={{ transition: "d 250ms" }}
        />
        {dangerTick && (
          <circle cx={dangerTick[0]} cy={dangerTick[1]} r="2.6" fill="var(--amber)" />
        )}
        <line
          x1="50"
          y1="50"
          x2={nx}
          y2={ny}
          stroke={hot ? "var(--red)" : "var(--text)"}
          strokeWidth="2"
          strokeLinecap="round"
          style={{ transition: "x2 250ms, y2 250ms" }}
        />
        <circle cx="50" cy="50" r="3.4" fill={hot ? "var(--red)" : "var(--text)"} />
      </svg>
      <div
        className="mono"
        style={{
          fontSize: "1.05rem",
          color: hot ? "var(--red)" : "var(--accent)",
          marginTop: "-0.4rem",
        }}
      >
        {value === undefined ? "—" : format(value)}
      </div>
      <div className="panel-label" style={{ marginTop: "0.15rem" }}>
        {label}
      </div>
    </div>
  );
}
