/* Reading the service's own OpenAPI document.
 *
 * The API section renders whatever /openapi.json says right now. Nothing
 * about the endpoints is written down in this app: FastAPI generates the
 * document from the same Pydantic models that validate the requests, so a
 * field added, renamed or removed on the server shows up here on the next
 * page load and cannot drift.
 */

export interface JSONSchema {
  $ref?: string;
  type?: string | string[];
  format?: string;
  title?: string;
  description?: string;
  default?: unknown;
  example?: unknown;
  examples?: unknown[];
  enum?: unknown[];
  const?: unknown;
  properties?: Record<string, JSONSchema>;
  required?: string[];
  items?: JSONSchema;
  additionalProperties?: JSONSchema | boolean;
  anyOf?: JSONSchema[];
  oneOf?: JSONSchema[];
  allOf?: JSONSchema[];
  prefixItems?: JSONSchema[];
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  minLength?: number;
  maxLength?: number;
  minItems?: number;
  maxItems?: number;
}

export interface Operation {
  method: "get" | "post";
  path: string;
  operationId?: string;
  summary?: string;
  description?: string;
  tags?: string[];
  requestSchema: JSONSchema | null;
  responseSchema: JSONSchema | null;
}

export interface OpenAPIDoc {
  openapi?: string;
  info?: { title?: string; version?: string; description?: string };
  paths?: Record<string, Record<string, unknown>>;
  components?: { schemas?: Record<string, JSONSchema> };
}

export interface ApiSurface {
  title: string;
  version: string;
  description: string;
  operations: Operation[];
  schemas: Record<string, JSONSchema>;
}

const METHODS = ["get", "post"] as const;

export function readSurface(doc: OpenAPIDoc): ApiSurface {
  const schemas = doc.components?.schemas ?? {};
  const operations: Operation[] = [];

  for (const [path, item] of Object.entries(doc.paths ?? {})) {
    for (const method of METHODS) {
      const op = item[method] as Record<string, unknown> | undefined;
      if (!op) continue;
      const body = op.requestBody as
        | { content?: Record<string, { schema?: JSONSchema }> }
        | undefined;
      const responses = op.responses as
        | Record<string, { content?: Record<string, { schema?: JSONSchema }> }>
        | undefined;
      operations.push({
        method,
        path,
        operationId: op.operationId as string | undefined,
        summary: op.summary as string | undefined,
        description: op.description as string | undefined,
        tags: op.tags as string[] | undefined,
        requestSchema: body?.content?.["application/json"]?.schema ?? null,
        responseSchema:
          responses?.["200"]?.content?.["application/json"]?.schema ??
          responses?.["201"]?.content?.["application/json"]?.schema ??
          null,
      });
    }
  }

  operations.sort(
    (a, b) => groupRank(group(a.path)) - groupRank(group(b.path)) || a.path.localeCompare(b.path),
  );

  return {
    title: doc.info?.title ?? "API",
    version: doc.info?.version ?? "",
    description: doc.info?.description ?? "",
    operations,
    schemas,
  };
}

/** The second path segment, which is how this service is already organised. */
export function group(path: string): string {
  const parts = path.split("/").filter(Boolean);
  if (parts[0] === "v1" && parts.length > 1) return parts[1];
  return parts[0] ?? "root";
}

/** Solving comes first because that is what the service is for. */
export const GROUP_ORDER = [
  "solve",
  "fit",
  "estimate",
  "domains",
  "diagnose",
  "toolkit",
  "response",
  "decompose",
  "dynamics",
  "examples",
  "health",
];

export function groupRank(g: string): number {
  const i = GROUP_ORDER.indexOf(g);
  return i === -1 ? GROUP_ORDER.length : i;
}

export const GROUP_TITLES: Record<string, string> = {
  solve: "Solve a problem",
  fit: "Fit to observations",
  diagnose: "Diagnose a series",
  domains: "Named domains",
  toolkit: "Toolkit readings",
  estimate: "Estimation",
  dynamics: "Dynamics",
  response: "Response",
  decompose: "Decomposition",
  examples: "Examples",
  health: "Service",
};

/** Follow a local $ref; anything else is returned unchanged. */
/**
 * FastAPI hands back the Python docstring verbatim, so it arrives with reST
 * double-backtick literals and hard-wrapped lines in it. Unwrap the paragraphs
 * and drop the markup — the text is the documentation, the punctuation is not.
 */
export function prose(text: string | undefined): string {
  if (!text) return "";
  return text
    .replace(/``([^`]+)``/g, "$1")
    .replace(/:func:|:class:|:meth:|:mod:/g, "")
    .replace(/[ \t]*\n(?![ \t]*\n)[ \t]*/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export function deref(schema: JSONSchema | null | undefined, schemas: Record<string, JSONSchema>): JSONSchema | null {
  if (!schema) return null;
  if (!schema.$ref) return schema;
  const name = schema.$ref.split("/").pop();
  if (!name) return schema;
  const target = schemas[name];
  return target ? { ...target, title: target.title ?? name } : schema;
}

/** A short human type for a schema node: "number", "list of number", "PricingPayload". */
export function typeName(schema: JSONSchema | null | undefined, schemas: Record<string, JSONSchema>): string {
  if (!schema) return "any";
  if (schema.$ref) return schema.$ref.split("/").pop() ?? "object";
  if (schema.const !== undefined) return JSON.stringify(schema.const);
  if (schema.enum) return schema.enum.map((v) => JSON.stringify(v)).join(" | ");
  const variants = schema.anyOf ?? schema.oneOf;
  if (variants) {
    const named = variants
      .filter((v) => !(v.type === "null"))
      .map((v) => typeName(v, schemas));
    const nullable = variants.some((v) => v.type === "null");
    return named.join(" | ") + (nullable ? " | null" : "");
  }
  if (schema.allOf && schema.allOf.length === 1) return typeName(schema.allOf[0], schemas);
  const t = Array.isArray(schema.type) ? schema.type.join(" | ") : schema.type;
  if (t === "array") {
    if (schema.prefixItems) return `[${schema.prefixItems.map((v) => typeName(v, schemas)).join(", ")}]`;
    return `list of ${typeName(schema.items, schemas)}`;
  }
  if (t === "object" && schema.additionalProperties && typeof schema.additionalProperties === "object") {
    return `map to ${typeName(schema.additionalProperties, schemas)}`;
  }
  if (t === "integer") return "integer";
  return t ?? "object";
}

/** Constraints worth printing beside a field. */
export function constraints(schema: JSONSchema): string[] {
  const out: string[] = [];
  const push = (label: string, v: unknown) => {
    if (v !== undefined && v !== null) out.push(`${label} ${v}`);
  };
  push("≥", schema.minimum);
  push("≤", schema.maximum);
  push(">", schema.exclusiveMinimum);
  push("<", schema.exclusiveMaximum);
  if (schema.minItems !== undefined) out.push(`at least ${schema.minItems} items`);
  if (schema.maxItems !== undefined) out.push(`at most ${schema.maxItems} items`);
  if (schema.default !== undefined) out.push(`default ${JSON.stringify(schema.default)}`);
  return out;
}

/**
 * A request body that will actually validate, built from the schema's own
 * defaults and types. Where the schema says nothing, the placeholder is
 * plainly a placeholder — never a plausible-looking measurement.
 */
export function sampleBody(schema: JSONSchema | null, schemas: Record<string, JSONSchema>, depth = 0): unknown {
  const node = deref(schema, schemas);
  if (!node || depth > 4) return null;
  if (node.default !== undefined) return node.default;
  if (node.example !== undefined) return node.example;
  if (node.const !== undefined) return node.const;
  if (node.enum && node.enum.length) return node.enum[0];

  const variants = node.anyOf ?? node.oneOf;
  if (variants) {
    const first = variants.find((v) => v.type !== "null") ?? variants[0];
    return sampleBody(first, schemas, depth + 1);
  }
  if (node.allOf && node.allOf.length) return sampleBody(node.allOf[0], schemas, depth + 1);

  const t = Array.isArray(node.type) ? node.type[0] : node.type;
  if (t === "object" || node.properties) {
    const out: Record<string, unknown> = {};
    const required = new Set(node.required ?? []);
    for (const [key, child] of Object.entries(node.properties ?? {})) {
      const childNode = deref(child, schemas);
      if (!required.has(key) && childNode?.default === undefined) continue;
      out[key] = sampleBody(child, schemas, depth + 1);
    }
    return out;
  }
  if (t === "array") {
    if (node.prefixItems) return node.prefixItems.map((v) => sampleBody(v, schemas, depth + 1));
    return [sampleBody(node.items ?? {}, schemas, depth + 1)];
  }
  if (t === "integer") return node.minimum ?? 1;
  if (t === "number") return node.minimum ?? 1.0;
  if (t === "boolean") return false;
  if (t === "null") return null;
  return "";
}
