"use client";

import { useEffect, useState } from "react";

/* A thin strip of live readings computed by the deployed backend the moment
   the page loads — proof the instruments are on, not screenshots. */

const RPS = {
  payoffs: [
    [
      [0, -1, 1],
      [1, 0, -1],
      [-1, 1, 0],
    ],
    [
      [0, 1, -1],
      [-1, 0, 1],
      [1, -1, 0],
    ],
  ],
  lam: 1.5,
};

interface Live {
  r?: number;
  epr?: number;
  db?: boolean;
  version?: string;
  state: "loading" | "ok" | "down";
}

export function LiveStrip() {
  const [live, setLive] = useState<Live>({ state: "loading" });

  useEffect(() => {
    let alive = true;
    const body = JSON.stringify(RPS);
    const opts = { method: "POST", headers: { "Content-Type": "application/json" }, body };
    Promise.all([
      fetch("/api/v1/health").then((r) => r.json()),
      fetch("/api/v1/response", opts).then((r) => r.json()),
      fetch("/api/v1/dynamics/stationary", opts).then((r) => r.json()),
    ])
      .then(([h, resp, dyn]) => {
        if (!alive) return;
        setLive({
          state: "ok",
          version: h.library,
          r: resp.reciprocity_defect,
          epr: dyn.epr,
          db: dyn.detailed_balance,
        });
      })
      .catch(() => alive && setLive({ state: "down" }));
    return () => {
      alive = false;
    };
  }, []);

  if (live.state === "down") {
    return (
      <div className="card" style={{ borderColor: "#7a5c22" }}>
        <span className="badge" data-tone="warn">
          backend unreachable
        </span>{" "}
        <span style={{ color: "var(--text-dim)", fontSize: "0.9rem" }}>
          Live readings are offline; everything else on this site is static and keeps working.
        </span>
      </div>
    );
  }

  return (
    <div className="meter-grid" aria-label="live readings from the deployed backend">
      <div className="card">
        <div className="panel-label">live · RPS at λ = 1.5</div>
        <div className="reading">{live.r === undefined ? "—" : `ℛ = ${live.r.toFixed(3)}`}</div>
        <p style={{ color: "var(--text-faint)", fontSize: "0.78rem", margin: "0.4rem 0 0" }}>
          reciprocity defect, computed just now
        </p>
      </div>
      <div className="card">
        <div className="panel-label">live · RPS entropy production</div>
        <div className="reading" data-tone="warn">
          {live.epr === undefined ? "—" : `σ = ${live.epr.toFixed(3)}`}
        </div>
        <p style={{ color: "var(--text-faint)", fontSize: "0.78rem", margin: "0.4rem 0 0" }}>
          {live.db === false ? "detailed balance broken — a driven system" : "nats per unit time"}
        </p>
      </div>
      <div className="card">
        <div className="panel-label">solver</div>
        <div className="reading" data-tone="neutral">
          {live.version ? `strataq ${live.version}` : "—"}
        </div>
        <p style={{ color: "var(--text-faint)", fontSize: "0.78rem", margin: "0.4rem 0 0" }}>
          float64 JAX, serving from a free-tier VM
        </p>
      </div>
    </div>
  );
}
