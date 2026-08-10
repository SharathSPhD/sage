"use client";

import { useEffect, useState } from "react";

export function HealthDot() {
  const [state, setState] = useState<"unknown" | "ok" | "down">("unknown");
  const [version, setVersion] = useState("");

  useEffect(() => {
    let alive = true;
    fetch("/api/v1/health")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((d) => {
        if (!alive) return;
        setState("ok");
        setVersion(d.library ?? "");
      })
      .catch(() => alive && setState("down"));
    return () => {
      alive = false;
    };
  }, []);

  return (
    <span className="status-dot" title={version ? `strataq ${version}` : "backend status"}>
      <span className="dot" data-state={state} />
      {state === "ok" ? "live" : state === "down" ? "offline" : "…"}
    </span>
  );
}
