"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

// Ordered by what a visitor came to do: solve one, browse the problem
// types, bring their own numbers, call the API directly, read the theory.
// The research programme lives one click deeper, under /research.
const LINKS = [
  { href: "/solve", label: "Solve" },
  { href: "/demos", label: "Demos" },
  { href: "/situations", label: "Problems" },
  { href: "/data", label: "Your data" },
  { href: "/api", label: "API" },
  { href: "/learn", label: "Learn" },
  { href: "/research", label: "Research" },
];

export function NavLinks() {
  const path = usePathname();
  return (
    <div className="links">
      {LINKS.map((l) => (
        <Link key={l.href} href={l.href} data-active={path === l.href || path.startsWith(`${l.href}/`)}>
          {l.label}
        </Link>
      ))}
    </div>
  );
}
