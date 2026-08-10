import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { HealthDot } from "./components/HealthDot";
import { NavLinks } from "./components/NavLinks";

export const metadata: Metadata = {
  title: "SAGE Labs — instruments for strategic systems",
  description:
    "Measurement instruments for stochastic strategic systems: susceptibility, reciprocity, dissipation, phase. Every reading calibrated on games where the answer is known.",
};

function BrandMark() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle cx="12" cy="12" r="10" stroke="var(--accent)" strokeWidth="1.5" />
      <path
        d="M12 3 L12 12 L18.4 16.4"
        stroke="var(--accent)"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <circle cx="12" cy="12" r="1.8" fill="var(--accent)" />
    </svg>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <nav className="site-nav">
          <Link href="/" className="brand">
            <BrandMark />
            SAGE Labs
          </Link>
          <NavLinks />
          <div style={{ marginLeft: "auto", display: "flex", gap: "1.2rem", alignItems: "center" }}>
            <HealthDot />
            <a
              href="https://github.com/SharathSPhD/sage"
              style={{ color: "var(--text-dim)", fontSize: "0.88rem" }}
            >
              GitHub
            </a>
          </div>
        </nav>
        <main>{children}</main>
        <footer className="site-footer">
          <span>
            Every number regenerates from fixed seeds in the{" "}
            <a href="https://github.com/SharathSPhD/sage">open repository</a>; adversarial reviews
            that closed each instrument are part of the public record.
          </span>
          <span>
            <a href="https://sharathsphd.github.io/sage/">docs</a>
            {" · "}
            <a href="https://sharathsphd.github.io/sage/progress/">gate dashboard</a>
            {" · "}
            <a href="/api/v1/health">API</a>
          </span>
        </footer>
      </body>
    </html>
  );
}
