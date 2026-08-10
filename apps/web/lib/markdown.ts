import { marked } from "marked";
import katex from "katex";

// Render theory markdown to HTML with LaTeX support. Math is stashed before
// marked runs (so underscores/asterisks inside TeX survive) and re-inserted
// after. throwOnError: false — a bad formula renders in red rather than
// killing the whole page build.
export function renderMarkdown(md: string): string {
  const stash: string[] = [];
  const keep = (html: string) => {
    stash.push(html);
    return `MATHSLOT${stash.length - 1}ENDSLOT`;
  };

  let src = md.replace(/\$\$([\s\S]+?)\$\$/g, (_m, tex: string) =>
    keep(katex.renderToString(tex, { displayMode: true, throwOnError: false })),
  );
  src = src.replace(/(?<![\\$])\$([^\n$]+?)\$/g, (_m, tex: string) =>
    keep(katex.renderToString(tex, { throwOnError: false })),
  );

  let html = marked.parse(src, { async: false }) as string;
  html = html.replace(/MATHSLOT(\d+)ENDSLOT/g, (_m, i: string) => stash[Number(i)]);
  return html;
}

// First non-heading paragraph, math stripped — used as the index-card teaser.
export function firstParagraph(md: string): string {
  const lines = md.split("\n");
  const out: string[] = [];
  let inPara = false;
  for (const line of lines) {
    const t = line.trim();
    if (t.startsWith("#") || t === "") {
      if (inPara) break;
      continue;
    }
    inPara = true;
    out.push(t);
  }
  return out
    .join(" ")
    .replace(/\$\$[\s\S]+?\$\$/g, "")
    .replace(/\$[^$\n]+\$/g, "…")
    .replace(/[*_`]/g, "")
    .slice(0, 220);
}
