import { marked } from "marked";
import { notFound } from "next/navigation";
import { getExplainer, listExplainers } from "../../../lib/theory";

export function generateStaticParams() {
  return listExplainers().map((e) => ({ slug: e.slug }));
}

export default async function ExplainerPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const explainer = getExplainer(slug);
  if (!explainer) notFound();
  const html = marked.parse(explainer.markdown) as string;
  return (
    <article
      style={{ lineHeight: 1.65 }}
      // Single-source markdown from docs/theory, trusted repo content.
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
