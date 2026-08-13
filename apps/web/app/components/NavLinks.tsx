"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

// Ordered by what a visitor came to do: solve one, browse the rest, race it,
// learn the idea. Everything about the research programme lives one click
// deeper, under /research.
const LINKS = [
  { href: "/solve", label: "Solve" },
  { href: "/situations", label: "Situations" },
  { href: "/play", label: "Play" },
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
