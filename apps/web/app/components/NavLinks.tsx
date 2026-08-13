"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

// Ordered by what a visitor came to do: solve one, pick a problem type, test a
// rule over repeated rounds, read the theory. The research programme lives one
// click deeper, under /research.
const LINKS = [
  { href: "/solve", label: "Solve" },
  { href: "/situations", label: "Problems" },
  { href: "/play", label: "Backtest" },
  { href: "/learn", label: "Learn" },
  { href: "/research", label: "Research" },
];

export function NavLinks() {
  const path = usePathname();
  return (
    <div className="links">
      {LINKS.map((l) => (
        <Link key={l.href} href={l.href} data-active={path.startsWith(l.href)}>
          {l.label}
        </Link>
      ))}
    </div>
  );
}
