import Link from "next/link";
import { notFound } from "next/navigation";
import "katex/dist/katex.min.css";
import { listExplainers } from "../../../lib/theory";
import { renderMarkdown } from "../../../lib/markdown";
import { ExplorablePanel } from "../../components/panels/registry";

export function generateStaticParams() {
  return listExplainers().map((e) => ({ slug: e.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const e = listExplainers().find((x) => x.slug === slug);
  return { title: e ? `${e.title} — SAGE Labs` : "Learn — SAGE Labs" };
}

export default async function ExplainerPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const explainers = listExplainers();
  const idx = explainers.findIndex((e) => e.slug === slug);
  if (idx === -1) notFound();
  const e = explainers[idx];
  const prev = idx > 0 ? explainers[idx - 1] : null;
  const next = idx < explainers.length - 1 ? explainers[idx + 1] : null;
  const html = renderMarkdown(e.markdown);

  return (
    <div className="wrap-narrow" style={{ paddingTop: "2.2rem" }}>
      <div className="panel-label">
        <Link href="/learn" style={{ color: "var(--text-faint)" }}>
          Learn
        </Link>{" "}
        · {String(idx + 1).padStart(2, "0")} / {String(explainers.length).padStart(2, "0")}
      </div>
      {/* html is rendered at build time from repo-authored markdown (docs/theory) — same trust domain as the code; no user-supplied content flows here */}
      <article className="prose" dangerouslySetInnerHTML={{ __html: html }} />
      <ExplorablePanel slug={slug} />
      <nav className="pager">
        {prev ? (
          <Link href={`/learn/${prev.slug}`}>
            <div className="card">
              <div className="dir">← previous</div>
              {prev.title}
            </div>
          </Link>
        ) : (
          <span />
        )}
        {next ? (
          <Link href={`/learn/${next.slug}`} style={{ textAlign: "right" }}>
            <div className="card">
              <div className="dir">next →</div>
              {next.title}
            </div>
          </Link>
        ) : (
          <Link href="/lab" style={{ textAlign: "right" }}>
            <div className="card">
              <div className="dir">now →</div>
              Open the Lab and watch the meters
            </div>
          </Link>
        )}
      </nav>
    </div>
  );
}
