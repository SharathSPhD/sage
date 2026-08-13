"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  // /diagnose is the front door: it is what a practitioner arriving with data
  // needs first, so it leads. /tools still works and now points here.
  { href: "/diagnose", label: "Diagnose" },
  { href: "/learn", label: "Learn" },
  { href: "/lab", label: "Lab" },
  { href: "/phase", label: "Phase map" },
  { href: "/network", label: "Network" },
  { href: "/markets", label: "Markets" },
  { href: "/findings", label: "Findings" },
  { href: "/tools", label: "Your data" },
  { href: "/blotto", label: "Blotto" },
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
