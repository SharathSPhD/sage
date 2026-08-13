"use client";

import { useEffect, useState } from "react";

/** Whether the solver behind every number on this site is answering. */
export function HealthDot() {
  const [state, setState] = useState<"loading" | "ok" | "down">("loading");
  const [version, setVersion] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    fetch("/api/v1/health")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((d: { library?: string }) => {
        if (!alive) return;
        setVersion(d.library ?? null);
        setState("ok");
      })
      .catch(() => alive && setState("down"));
    return () => {
      alive = false;
    };
  }, []);

  return (
    <span className="status-dot" title={state === "ok" ? "The solver is answering." : "The solver is not answering."}>
      <span className="dot" data-state={state} />
      {state === "ok" ? `strataq ${version ?? "live"}` : state === "down" ? "solver offline" : "…"}
    </span>
  );
}
