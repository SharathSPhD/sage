"use client";

/* A technical name, kept out of the way.
 *
 * The rule from the product spec is that no Greek letter appears in primary
 * copy. It still has to be reachable — a researcher who wants to know which
 * parameter a control is should not have to guess — so it lives here.
 *
 * WCAG 1.4.13 applies to anything revealed on hover or focus: the definition
 * has to be hoverable (it is inside the trigger's container), persistent
 * (it stays until focus or the pointer leaves) and dismissible without moving
 * either — which is what the Escape handler is for.
 */

import { useId, useState } from "react";

export function Term({ term, explain }: { term: string; explain: string }) {
  const id = useId();
  const [dismissed, setDismissed] = useState(false);

  return (
    <span
      className="term"
      data-dismissed={dismissed}
      onMouseEnter={() => setDismissed(false)}
      onKeyDown={(e) => {
        if (e.key === "Escape") setDismissed(true);
      }}
    >
      <button
        type="button"
        className="term-trigger"
        aria-describedby={id}
        onFocus={() => setDismissed(false)}
      >
        {term}
      </button>
      <span className="term-body" id={id} role="note">
        {explain}
      </span>
    </span>
  );
}
