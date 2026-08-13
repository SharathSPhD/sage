import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { HealthDot } from "./components/HealthDot";
import { NavLinks } from "./components/NavLinks";

export const metadata: Metadata = {
  title: "SAGE — decide when the other side is deciding too",
  description:
    "Put in the numbers you already have and get the move to make, what the other side is likely to do, what it is worth, and what would change the answer.",
};

function BrandMark() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle cx="12" cy="12" r="10" stroke="var(--accent)" strokeWidth="1.5" />
      <path d="M12 3 L12 12 L18.4 16.4" stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round" />
      <circle cx="12" cy="12" r="1.8" fill="var(--accent)" />
    </svg>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <a href="#main" className="skip-link">
          Skip to the answer
        </a>
        <nav className="site-nav">
          <Link href="/" className="brand">
            <BrandMark />
            SAGE
          </Link>
          <NavLinks />
          <div style={{ marginLeft: "auto", display: "flex", gap: "1.2rem", alignItems: "center" }}>
            <HealthDot />
            <a href="https://github.com/SharathSPhD/sage" style={{ color: "var(--text-dim)", fontSize: "0.88rem" }}>
              GitHub
            </a>
          </div>
        </nav>
        <main id="main">{children}</main>
        <footer className="site-footer">
          <span>
            Numbers marked illustrative are a plausible stand-in and say so on the page. Everything else regenerates
            from fixed seeds in <a href="https://github.com/SharathSPhD/sage">the open repository</a>.
          </span>
          <span>
            <Link href="/research">how it works</Link>
            {" · "}
            <a href="https://sharathsphd.github.io/sage/">docs</a>
            {" · "}
            <a href="/api/v1/health">API</a>
          </span>
        </footer>
      </body>
    </html>
  );
}
