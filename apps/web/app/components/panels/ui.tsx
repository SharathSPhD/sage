"use client";

/* Shared pieces for the explorable panels. Every panel declares its compute
   provenance: client-side math is instant but tethered to library goldens;
   live-solver readings carry the float64 badge. */

export function PanelShell({
  title,
  provenance,
  children,
}: {
  title: string;
  provenance: "client" | "live";
  children: React.ReactNode;
}) {
  return (
    <section className="card explorable" aria-label={title}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: "1rem" }}>
        <div className="panel-label" style={{ marginBottom: 0 }}>
          ▶ try it · {title}
        </div>
        <span className="badge" data-tone={provenance === "live" ? "ok" : undefined}>
          {provenance === "live" ? "live float64 solver" : "in-browser · goldens-checked"}
        </span>
      </div>
      <div style={{ marginTop: "0.9rem" }}>{children}</div>
    </section>
  );
}

/** Horizontal bar chart with animated widths and a shared scale. */
export function Bars({
  values,
  labels,
  color = "var(--accent)",
  format = (v: number) => v.toFixed(2),
  max,
}: {
  values: number[];
  labels: string[];
  color?: string;
  format?: (v: number) => string;
  max?: number;
}) {
  const lo = Math.min(0, ...values);
  const hi = max ?? Math.max(...values.map((v) => Math.abs(v)), 1e-9);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {values.map((v, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span className="mono" style={{ width: "4.2rem", fontSize: "0.7rem", color: "var(--text-faint)", textAlign: "right" }}>
            {labels[i]}
          </span>
          <div style={{ flex: 1, height: 16, background: "var(--bg-raised)", borderRadius: 3, position: "relative", overflow: "hidden" }}>
            <div
              style={{
                position: "absolute",
                left: lo < 0 ? "50%" : 0,
                width: `${(Math.abs(v) / (hi || 1)) * (lo < 0 ? 50 : 100)}%`,
                transform: lo < 0 && v < 0 ? "translateX(-100%)" : undefined,
                top: 0,
                bottom: 0,
                background: color,
                borderRadius: 3,
                transition: "width 160ms ease-out",
              }}
            />
          </div>
          <span className="mono" style={{ width: "3.6rem", fontSize: "0.72rem", color: "var(--text-dim)" }}>
            {format(v)}
          </span>
        </div>
      ))}
    </div>
  );
}

/** λ slider on a log scale, with an ∞ (Nash) stop at the right end. */
export function LambdaSlider({
  lam,
  setLam,
  min = 0.01,
  max = 50,
  infinity = false,
  label = "rationality λ",
}: {
  lam: number; // Infinity allowed when infinity=true
  setLam: (v: number) => void;
  min?: number;
  max?: number;
  infinity?: boolean;
  label?: string;
}) {
  const span = Math.log(max / min);
  const toT = (l: number) => (l === Infinity ? 1 : Math.min(1, Math.log(l / min) / span) * (infinity ? 0.96 : 1));
  const fromT = (t: number) => {
    if (infinity && t > 0.96) return Infinity;
    return min * Math.exp((t / (infinity ? 0.96 : 1)) * span);
  };
  return (
    <div>
      <div className="panel-label">
        {label} ={" "}
        <span style={{ color: "var(--accent)" }}>{lam === Infinity ? "∞ (Nash)" : lam.toPrecision(3)}</span>
      </div>
      <input
        type="range"
        min={0}
        max={1}
        step={0.002}
        value={toT(lam)}
        onChange={(e) => setLam(fromT(Number(e.target.value)))}
        aria-label={label}
      />
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.66rem", color: "var(--text-faint)", fontFamily: "var(--mono)" }}>
        <span>{min} · noise</span>
        <span>{infinity ? "∞ · Nash" : `${max} · sharp`}</span>
      </div>
    </div>
  );
}

export function NumberDial({
  value,
  setValue,
  min,
  max,
  step,
  label,
  format = (v: number) => v.toFixed(2),
}: {
  value: number;
  setValue: (v: number) => void;
  min: number;
  max: number;
  step: number;
  label: string;
  format?: (v: number) => string;
}) {
  return (
    <div>
      <div className="panel-label">
        {label} = <span style={{ color: "var(--accent)" }}>{format(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => setValue(Number(e.target.value))}
        aria-label={label}
      />
    </div>
  );
}
