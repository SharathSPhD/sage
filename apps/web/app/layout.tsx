import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { HealthDot } from "./components/HealthDot";
import { NavLinks } from "./components/NavLinks";
import { ThemeToggle } from "./components/ThemeToggle";

export const metadata: Metadata = {
  title: {
    default: "SAGE — solve pricing, auction, electricity, routing and allocation problems",
    template: "%s — SAGE",
  },
  description:
    "Enter your costs, grid and demand model; get the price, bid, offer, split or toll to set, what it earns, and how it moves when an input does.",
};

// Stamps the stored theme onto <html> before first paint, so a returning
// visitor never sees the other theme flash. Kept to one expression.
const THEME_BOOT = `try{var t=localStorage.getItem('sage-theme');if(t==='dark'||t==='light')document.documentElement.setAttribute('data-theme',t)}catch(e){}`;

function BrandMark() {
  return (
    <svg width="21" height="21" viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle cx="12" cy="12" r="9.2" stroke="var(--accent)" strokeWidth="1.6" />
      <path d="M12 3.4V12l6.1 4.1" stroke="var(--accent)" strokeWidth="1.6" strokeLinecap="round" />
      <circle cx="12" cy="12" r="1.9" fill="var(--accent)" />
    </svg>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOT }} />
      </head>
      <body>
        <a href="#main" className="skip-link">
          Skip to the answer
        </a>
        <nav className="site-nav" aria-label="Primary">
          <Link href="/" className="brand">
            <BrandMark />
            SAGE
          </Link>
          <NavLinks />
          <div className="nav-end">
            <HealthDot />
            <ThemeToggle />
          </div>
        </nav>
        <main id="main">{children}</main>
        <footer className="site-footer">
          <div className="footer-inner">
            <div>
              <p className="footer-blurb">
                <strong>SAGE</strong> solves the problems where the other side is optimising too — pricing, tenders,
                electricity offers, traffic assignment and budget allocation — over one solver, served by the{" "}
                <code>strataq</code> API.
              </p>
            </div>
            <div>
              <h3>Solve</h3>
              <ul>
                <li><Link href="/solve">Studio</Link></li>
                <li><Link href="/situations">Problem types</Link></li>
                <li><Link href="/data">Bring your data</Link></li>
                <li><Link href="/play">Backtest</Link></li>
              </ul>
            </div>
            <div>
              <h3>Build</h3>
              <ul>
                <li><Link href="/api">API console</Link></li>
                <li><a href="/api/openapi.json">openapi.json</a></li>
                <li><a href="https://sharathsphd.github.io/sage/">Docs</a></li>
                <li><a href="https://github.com/SharathSPhD/sage">Source</a></li>
              </ul>
            </div>
            <div>
              <h3>Read</h3>
              <ul>
                <li><Link href="/learn">Explainers</Link></li>
                <li><Link href="/research">Research</Link></li>
                <li><Link href="/findings">Findings</Link></li>
              </ul>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
