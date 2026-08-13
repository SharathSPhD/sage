"use client";

/* A magnitude shade that cannot break text contrast.
 *
 * A hard sequential ramp gets dark enough at the top that any single text
 * colour fails against one end or the other, and the ramp's own steps differ
 * between the light and dark themes. So the fill is one hue laid over the
 * surface at an opacity proportional to the value, capped where the surface
 * still carries `--text` at better than 6:1 in both themes. The mark is a
 * separate layer from the label, so the label never inherits the opacity.
 */

export const SHADE_MAX = 0.55;

export function Shade({ t }: { t: number }) {
  const clamped = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
  return (
    <i
      aria-hidden
      style={{
        position: "absolute",
        inset: 0,
        borderRadius: "inherit",
        background: "var(--series-1)",
        opacity: SHADE_MAX * clamped,
        transition: "opacity var(--dur-2) var(--ease)",
      }}
    />
  );
}

/** The scale key, as the same shades the cells use. */
export function ShadeLegend({ low = "less", high = "more" }: { low?: string; high?: string }) {
  return (
    <span className="blotto-legend">
      {low}
      <span className="swatches">
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <i
            key={i}
            style={{
              position: "relative",
              background: "var(--surface-2)",
              overflow: "hidden",
            }}
          >
            <i
              style={{
                position: "absolute",
                inset: 0,
                background: "var(--series-1)",
                opacity: (SHADE_MAX * i) / 5,
              }}
            />
          </i>
        ))}
      </span>
      {high}
    </span>
  );
}
