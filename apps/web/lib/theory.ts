import fs from "node:fs";
import path from "node:path";

// SINGLE SOURCE RULE: Learn prose lives in docs/theory (rendered by the docs
// site too); this module only reads it. Never author explainer text here.
const THEORY_DIR = path.join(process.cwd(), "content", "theory");

export interface Explainer {
  slug: string;
  title: string;
  markdown: string;
}

export function listExplainers(): Explainer[] {
  return fs
    .readdirSync(THEORY_DIR)
    .filter((f) => /^\d\d-.*\.md$/.test(f))
    .sort()
    .map((f) => {
      const markdown = fs.readFileSync(path.join(THEORY_DIR, f), "utf-8");
      const title = markdown.split("\n")[0].replace(/^#\s*/, "");
      return { slug: f.replace(/\.md$/, ""), title, markdown };
    });
}

export function getExplainer(slug: string): Explainer | undefined {
  return listExplainers().find((e) => e.slug === slug);
}
