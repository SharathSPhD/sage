"use client";

/* A schema rendered as a field list: name, type, constraints, and whatever
 * docstring the server put on it. Nested models expand inline one level at a
 * time, so a big response reads top-down instead of as a wall.
 */

import { useState } from "react";
import { constraints, deref, prose, typeName, type JSONSchema } from "../../lib/openapi";

export function SchemaTable({
  schema,
  schemas,
  depth = 0,
}: {
  schema: JSONSchema | null;
  schemas: Record<string, JSONSchema>;
  depth?: number;
}) {
  const node = deref(schema, schemas);
  if (!node) return <p className="figure-note">No body.</p>;

  if (!node.properties) {
    return (
      <div className="schema-tree">
        <div className="schema-row">
          <span className="schema-name">(body)</span>
          <span className="schema-type">{typeName(node, schemas)}</span>
          <span className="schema-doc">{prose(node.description) || "Shape defined by the endpoint."}</span>
        </div>
      </div>
    );
  }

  const required = new Set(node.required ?? []);
  const entries = Object.entries(node.properties);

  return (
    <div className="schema-tree">
      {entries.map(([name, child]) => (
        <Row
          key={name}
          name={name}
          child={child}
          required={required.has(name)}
          schemas={schemas}
          depth={depth}
        />
      ))}
    </div>
  );
}

function Row({
  name,
  child,
  required,
  schemas,
  depth,
}: {
  name: string;
  child: JSONSchema;
  required: boolean;
  schemas: Record<string, JSONSchema>;
  depth: number;
}) {
  const [open, setOpen] = useState(false);
  const resolved = deref(child, schemas);
  const inner = resolved?.items ? deref(resolved.items, schemas) : null;
  const expandable =
    depth < 2 && ((resolved?.properties && Object.keys(resolved.properties).length > 0) || (inner?.properties && Object.keys(inner.properties).length > 0));
  const bits = resolved ? constraints(resolved) : [];

  return (
    <>
      <div className="schema-row">
        <span className="schema-name">
          {name}
          {required && (
            <span className="req" title="required">
              *
            </span>
          )}
        </span>
        <span className="schema-type">
          {typeName(child, schemas)}
          {expandable && (
            <button
              type="button"
              data-quiet="true"
              style={{ marginLeft: "0.4rem", padding: "0 0.3rem", minHeight: "auto", fontSize: "0.7rem" }}
              aria-expanded={open}
              onClick={() => setOpen((o) => !o)}
            >
              {open ? "hide fields" : "fields"}
            </button>
          )}
        </span>
        <span className="schema-doc">
          {prose(resolved?.description)}
          {bits.length > 0 && (
            <span style={{ color: "var(--text-3)", fontFamily: "var(--mono)", fontSize: "0.75rem" }}>
              {resolved?.description ? " · " : ""}
              {bits.join(" · ")}
            </span>
          )}
        </span>
      </div>
      {open && (
        <div className="schema-nested">
          <SchemaTable schema={inner?.properties ? inner : resolved ?? null} schemas={schemas} depth={depth + 1} />
        </div>
      )}
    </>
  );
}
