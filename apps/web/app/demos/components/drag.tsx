"use client";

import { useCallback, useRef, useState } from "react";

export interface DragNumberOptions {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  /** Which pointer axis drives the value. "y" is inverted: up increases. */
  axis?: "x" | "y";
  /** Pixels of travel for the full min..max range. */
  travelPx?: number;
  label: string;
  valueText?: (v: number) => string;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/**
 * Makes an SVG or HTML element the handle for one number: pointer drag for the
 * mouse and touch, arrow keys for everyone else, and the ARIA slider contract so
 * the value is announced either way.
 */
export function useDragNumber(opts: DragNumberOptions) {
  const { value, min, max, step, onChange, axis = "y", travelPx = 220, label, valueText } = opts;
  const [dragging, setDragging] = useState(false);
  const start = useRef({ p: 0, v: 0 });

  const quantise = useCallback(
    (v: number) => {
      const snapped = Math.round(v / step) * step;
      return clamp(Number(snapped.toFixed(6)), min, max);
    },
    [step, min, max],
  );

  const onPointerDown = useCallback(
    (e: React.PointerEvent) => {
      if (e.button !== undefined && e.button !== 0) return;
      (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
      start.current = { p: axis === "y" ? e.clientY : e.clientX, v: value };
      setDragging(true);
      e.preventDefault();
    },
    [axis, value],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragging) return;
      const now = axis === "y" ? e.clientY : e.clientX;
      const delta = axis === "y" ? start.current.p - now : now - start.current.p;
      onChange(quantise(start.current.v + (delta / travelPx) * (max - min)));
    },
    [dragging, axis, onChange, quantise, travelPx, max, min],
  );

  const end = useCallback(() => setDragging(false), []);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      const big = (max - min) / 10;
      let next: number | null = null;
      if (e.key === "ArrowUp" || e.key === "ArrowRight") next = value + step;
      else if (e.key === "ArrowDown" || e.key === "ArrowLeft") next = value - step;
      else if (e.key === "PageUp") next = value + big;
      else if (e.key === "PageDown") next = value - big;
      else if (e.key === "Home") next = min;
      else if (e.key === "End") next = max;
      if (next === null) return;
      e.preventDefault();
      onChange(quantise(next));
    },
    [value, step, min, max, onChange, quantise],
  );

  return {
    dragging,
    handleProps: {
      role: "slider" as const,
      tabIndex: 0,
      "aria-label": label,
      "aria-valuemin": min,
      "aria-valuemax": max,
      "aria-valuenow": Number(value.toFixed(4)),
      "aria-valuetext": valueText ? valueText(value) : undefined,
      "aria-orientation": (axis === "y" ? "vertical" : "horizontal") as "vertical" | "horizontal",
      onPointerDown,
      onPointerMove,
      onPointerUp: end,
      onPointerCancel: end,
      onLostPointerCapture: end,
      onKeyDown,
      style: { cursor: axis === "y" ? "ns-resize" : "ew-resize", touchAction: "none" as const },
    },
  };
}
