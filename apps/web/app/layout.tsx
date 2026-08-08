import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "SAGE Labs",
  description:
    "Measurement instruments for stochastic strategic systems: susceptibility, reciprocity, dissipation, phase.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body
        style={{
          fontFamily: "system-ui, sans-serif",
          margin: 0,
          color: "#1a202c",
          background: "#fafaf7",
        }}
      >
        <nav
          style={{
            display: "flex",
            gap: "1.5rem",
            padding: "0.9rem 2rem",
            borderBottom: "1px solid #e2e2dc",
            background: "#0f3d3e",
            color: "white",
            alignItems: "baseline",
          }}
        >
          <Link href="/" style={{ color: "white", fontWeight: 700, textDecoration: "none" }}>
            SAGE Labs
          </Link>
          <Link href="/learn" style={{ color: "#cde5e0", textDecoration: "none" }}>
            Learn
          </Link>
          <Link href="/lab" style={{ color: "#cde5e0", textDecoration: "none" }}>
            Lab
          </Link>
          <a
            href="https://github.com/SharathSPhD/sage"
            style={{ marginLeft: "auto", color: "#cde5e0", textDecoration: "none", fontSize: "0.9rem" }}
          >
            code + gates + paper
          </a>
        </nav>
        <main style={{ maxWidth: "52rem", margin: "0 auto", padding: "2rem 1.5rem" }}>{children}</main>
      </body>
    </html>
  );
}
