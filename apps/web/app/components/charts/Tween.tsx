"use client";

/* A number that glides to its new value instead of snapping. Used for every
 * headline figure, so a drag reads as one quantity moving rather than a
 * flicker of unrelated numbers.
 */

import { useTween } from "../../../lib/anim";

export function TweenNumber({
  value,
  format,
  ms = 380,
}: {
  value: number;
  format: (v: number) => string;
  ms?: number;
}) {
  const shown = useTween(Number.isFinite(value) ? value : 0, ms);
  return (
    <span className="tween" aria-live="off">
      {format(shown)}
    </span>
  );
}
