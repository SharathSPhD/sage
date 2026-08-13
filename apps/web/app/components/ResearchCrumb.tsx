import Link from "next/link";

/* These pages used to be in the top navigation. They are evidence now, not
 * front doors, and a visitor who lands on one from a search result needs a way
 * back to the rest of the evidence — and to the pages that produce answers. */

export function ResearchCrumb({ children }: { children?: React.ReactNode }) {
  return (
    <p className="crumb research-crumb">
      <Link href="/research">How it works, and how we know</Link>
      {children ? <span className="research-crumb-note"> · {children}</span> : null}
    </p>
  );
}
