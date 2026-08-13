"use client";

/* The API section.
 *
 * Everything here is read from the service's own /openapi.json at page load:
 * the endpoint list, the summaries, the request and response schemas. There
 * is no local copy of the contract to fall out of date — FastAPI generates
 * that document from the same models that validate the requests.
 */

import { useEffect, useMemo, useState } from "react";
import { CopyButton } from "../components/CopyButton";
import { SchemaTable } from "../components/SchemaTable";
import { EXAMPLE_BODIES, PYTHON_EQUIVALENT } from "../../lib/api-examples";
import {
  GROUP_TITLES,
  group,
  prose,
  readSurface,
  sampleBody,
  type ApiSurface,
  type Operation,
} from "../../lib/openapi";

const PUBLIC_BASE = "http://150.136.84.2";

export function ApiConsole() {
  const [surface, setSurface] = useState<ApiSurface | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string>("POST /v1/solve/pricing");

  useEffect(() => {
    let alive = true;
    fetch("/api/openapi.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((doc) => {
        if (!alive) return;
        const read = readSurface(doc);
        setSurface(read);
        if (!read.operations.some((o) => key(o) === selected) && read.operations.length) {
          setSelected(key(read.operations[0]));
        }
      })
      .catch((e: Error) => alive && setError(e.message));
    return () => {
      alive = false;
    };
    // The document is fetched once; `selected` is only read for a fallback.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const groups = useMemo(() => {
    if (!surface) return [];
    const map = new Map<string, Operation[]>();
    for (const op of surface.operations) {
      const g = group(op.path);
      if (!map.has(g)) map.set(g, []);
      map.get(g)!.push(op);
    }
    return [...map.entries()];
  }, [surface]);

  const current = surface?.operations.find((o) => key(o) === selected) ?? null;

  return (
    <div className="wrap page">
      <h1 className="surface-title">API</h1>
      <p className="surface-lede">
        Every answer on this site is one HTTP call. This page is generated from the service&apos;s own{" "}
        <a href="/api/openapi.json">openapi.json</a> each time it loads, so the schemas below are the ones the server is
        validating against right now.
      </p>

      <section className="card" style={{ marginBottom: "1.5rem" }}>
        <div className="knob-grid">
          <div>
            <div className="panel-label">Base URL</div>
            <p className="model-line" style={{ margin: 0 }}>
              {PUBLIC_BASE}
            </p>
            <p className="figure-note">
              Public, no key, permissive CORS. This app proxies it at <code>/api/*</code> so an HTTPS page can call an
              HTTP service.
            </p>
          </div>
          <div>
            <div className="panel-label">Library</div>
            <div className="reading" style={{ fontSize: "var(--text-lg)" }}>
              {surface ? `${surface.title} ${surface.version}` : "…"}
            </div>
            <p className="figure-note">The version string the service reports for itself.</p>
          </div>
          <div>
            <div className="panel-label">Errors</div>
            <p className="figure-note" style={{ marginTop: 0 }}>
              <code>422</code> with <code>{`{"detail": "…"}`}</code> for a body the library rejects, <code>413</code>{" "}
              when a grid or table is larger than the sync service allows, <code>503</code> when a dataset cannot be
              fetched. A solve that misses tolerance returns <code>200</code> with{" "}
              <code>success: false</code> and says why in <code>warnings</code>.
            </p>
          </div>
          <div>
            <div className="panel-label">Limits</div>
            <p className="figure-note" style={{ marginTop: 0 }}>
              Synchronous service: at most 12 actions a player, 3 players, 60 grid levels, 20,000 joint profiles, 200
              network edges and 20 origin-destination pairs. Anything larger belongs in the Python library.
            </p>
          </div>
        </div>
      </section>

      {error && (
        <p className="studio-error" role="alert">
          Could not read the API description: {error}. The service may be restarting; the endpoints themselves are
          unaffected.
        </p>
      )}

      <div className="api-layout">
        <nav className="api-index" aria-label="Endpoints">
          {groups.map(([g, ops]) => (
            <div key={g}>
              <h2>{GROUP_TITLES[g] ?? g}</h2>
              <ul>
                {ops.map((op) => (
                  <li key={key(op)}>
                    <button type="button" data-on={key(op) === selected} onClick={() => setSelected(key(op))}>
                      <span className="method-pill" data-m={op.method}>
                        {op.method.toUpperCase()}
                      </span>
                      <span className="path">{op.path}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}
          {!surface && !error && <p className="figure-note">Reading the API description…</p>}
        </nav>

        <div>{current && surface ? <EndpointPanel op={current} surface={surface} /> : null}</div>
      </div>
    </div>
  );
}

function key(op: Operation): string {
  return `${op.method.toUpperCase()} ${op.path}`;
}

function EndpointPanel({ op, surface }: { op: Operation; surface: ApiSurface }) {
  const id = key(op);
  const initial = useMemo(() => {
    const example = EXAMPLE_BODIES[id];
    const value = example ?? (op.requestSchema ? sampleBody(op.requestSchema, surface.schemas) : null);
    return value === null ? "" : JSON.stringify(value, null, 2);
  }, [id, op.requestSchema, surface.schemas]);

  const [body, setBody] = useState(initial);
  const [result, setResult] = useState<{ status: number; ms: number; text: string } | null>(null);
  const [sending, setSending] = useState(false);
  const [snippet, setSnippet] = useState<"curl" | "python">("curl");

  useEffect(() => {
    setBody(initial);
    setResult(null);
  }, [initial]);

  const send = async () => {
    setSending(true);
    const started = performance.now();
    try {
      const init: RequestInit = { method: op.method.toUpperCase() };
      if (op.method === "post") {
        init.headers = { "Content-Type": "application/json" };
        init.body = body.trim() === "" ? "{}" : body;
      }
      const response = await fetch(`/api${op.path}`, init);
      const text = await response.text();
      let pretty = text;
      try {
        pretty = JSON.stringify(JSON.parse(text), null, 2);
      } catch {
        /* not JSON — show it as it came */
      }
      setResult({ status: response.status, ms: Math.round(performance.now() - started), text: pretty });
    } catch (e) {
      setResult({ status: 0, ms: Math.round(performance.now() - started), text: String((e as Error).message ?? e) });
    } finally {
      setSending(false);
    }
  };

  const curl =
    op.method === "get"
      ? `curl ${PUBLIC_BASE}${op.path}`
      : `curl -X POST ${PUBLIC_BASE}${op.path} \\\n  -H 'Content-Type: application/json' \\\n  -d '${compact(body)}'`;

  const python =
    op.method === "get"
      ? `import requests\n\nr = requests.get("${PUBLIC_BASE}${op.path}", timeout=30)\nr.raise_for_status()\nprint(r.json())`
      : `import requests\n\nbody = ${pythonLiteral(body)}\n\nr = requests.post("${PUBLIC_BASE}${op.path}", json=body, timeout=30)\nr.raise_for_status()\nprint(r.json())`;

  const library = PYTHON_EQUIVALENT[id];

  return (
    <div className="api-endpoint">
      <div>
        <div className="api-endpoint-head">
          <span className="method-pill" data-m={op.method}>
            {op.method.toUpperCase()}
          </span>
          <h1>{op.path}</h1>
        </div>
        {op.summary && <p className="api-summary">{prose(op.summary)}</p>}
        {op.description && prose(op.description) !== prose(op.summary) && (
          <p className="api-desc">{prose(op.description)}</p>
        )}
      </div>

      <section className="card">
        <h3>Request body</h3>
        <SchemaTable schema={op.requestSchema} schemas={surface.schemas} />
      </section>

      <section className="card">
        <h3>Try it</h3>
        <div className="try-console">
          <div>
            <div className="code-head">
              <span className="panel-label">Body</span>
              <div className="row" style={{ gap: "0.4rem" }}>
                <button type="button" className="copy-btn" onClick={() => setBody(initial)}>
                  Reset
                </button>
                <CopyButton text={body} />
              </div>
            </div>
            <textarea
              value={body}
              onChange={(e) => setBody(e.target.value)}
              spellCheck={false}
              aria-label={`Request body for ${op.method.toUpperCase()} ${op.path}`}
              disabled={op.method === "get"}
              placeholder={op.method === "get" ? "This endpoint takes no body." : ""}
            />
            <div className="controls-actions" style={{ marginTop: "0.8rem", paddingTop: 0, borderTop: "none" }}>
              <button type="button" data-primary="true" onClick={send} disabled={sending}>
                {sending ? "Sending…" : `Send ${op.method.toUpperCase()}`}
              </button>
              <span className="figure-note" style={{ margin: 0 }}>
                Goes to the live service through this site&apos;s proxy.
              </span>
            </div>
          </div>
          <div>
            <div className="code-head">
              <span className="panel-label">Response</span>
              {result && (
                <span className="try-meta">
                  <span className="status-code" data-ok={result.status >= 200 && result.status < 300}>
                    {result.status === 0 ? "network error" : result.status}
                  </span>
                  <span>{result.ms} ms</span>
                  <span>{new Blob([result.text]).size.toLocaleString()} bytes</span>
                </span>
              )}
            </div>
            <pre className="try-out" aria-live="polite">
              {result ? result.text : "Send the request to see the response."}
            </pre>
          </div>
        </div>
      </section>

      <section className="card">
        <div className="code-head">
          <div className="snippet-tabs" role="group" aria-label="Snippet language">
            <button type="button" data-on={snippet === "curl"} onClick={() => setSnippet("curl")} aria-pressed={snippet === "curl"}>
              curl
            </button>
            <button
              type="button"
              data-on={snippet === "python"}
              onClick={() => setSnippet("python")}
              aria-pressed={snippet === "python"}
            >
              Python
            </button>
          </div>
          <CopyButton text={snippet === "curl" ? curl : python} />
        </div>
        <pre className="code-block">{snippet === "curl" ? curl : python}</pre>
        {library && (
          <>
            <div className="code-head" style={{ marginTop: "1rem" }}>
              <span className="panel-label">The same thing with the library</span>
              <CopyButton text={library} />
            </div>
            <pre className="code-block">{library}</pre>
            <p className="figure-note">
              <code>pip install strataq</code>. The library returns the same solution object this endpoint serialises.
            </p>
          </>
        )}
      </section>

      {op.responseSchema && (
        <section className="card">
          <h3>Response</h3>
          <SchemaTable schema={op.responseSchema} schemas={surface.schemas} />
          <p className="figure-note">
            Where the service declares only a generic object, send the request above — the response pane shows every
            field the endpoint actually returns.
          </p>
        </section>
      )}
    </div>
  );
}

function compact(body: string): string {
  try {
    return JSON.stringify(JSON.parse(body));
  } catch {
    return body.replace(/\s+/g, " ").trim();
  }
}

/** JSON is nearly Python; only the three literals differ. */
function pythonLiteral(body: string): string {
  try {
    const value = JSON.parse(body);
    return JSON.stringify(value, null, 4)
      .replace(/\btrue\b/g, "True")
      .replace(/\bfalse\b/g, "False")
      .replace(/\bnull\b/g, "None");
  } catch {
    return "{}";
  }
}
