"use client";

/* Light is the default. The toggle writes an explicit choice to
 * localStorage and stamps <html data-theme>, which wins over the OS
 * setting in both directions; with no stored choice the OS setting
 * decides via prefers-color-scheme in globals.css.
 */

import { useEffect, useState } from "react";

type Choice = "light" | "dark" | "system";

const NEXT: Record<Choice, Choice> = { system: "dark", dark: "light", light: "system" };

const LABEL: Record<Choice, string> = {
  system: "Theme: follow system. Switch to dark.",
  dark: "Theme: dark. Switch to light.",
  light: "Theme: light. Follow the system setting.",
};

function apply(choice: Choice) {
  const root = document.documentElement;
  if (choice === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", choice);
  try {
    if (choice === "system") window.localStorage.removeItem("sage-theme");
    else window.localStorage.setItem("sage-theme", choice);
  } catch {
    /* private mode; the stamp on <html> still holds for this page */
  }
}

export function ThemeToggle() {
  const [choice, setChoice] = useState<Choice>("system");

  useEffect(() => {
    const stored = document.documentElement.getAttribute("data-theme");
    setChoice(stored === "dark" || stored === "light" ? stored : "system");
  }, []);

  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={() => {
        const next = NEXT[choice];
        setChoice(next);
        apply(next);
      }}
      aria-label={LABEL[choice]}
      title={LABEL[choice]}
    >
      {choice === "dark" ? <Moon /> : choice === "light" ? <Sun /> : <Auto />}
    </button>
  );
}

function Sun() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
      <circle cx="12" cy="12" r="4.2" />
      <path d="M12 2.5v2.2M12 19.3v2.2M4.2 4.2l1.6 1.6M18.2 18.2l1.6 1.6M2.5 12h2.2M19.3 12h2.2M4.2 19.8l1.6-1.6M18.2 5.8l1.6-1.6" strokeLinecap="round" />
    </svg>
  );
}

function Moon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
      <path d="M20 14.2A8.2 8.2 0 0 1 9.8 4a8.2 8.2 0 1 0 10.2 10.2z" strokeLinejoin="round" />
    </svg>
  );
}

function Auto() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
      <circle cx="12" cy="12" r="8.2" />
      <path d="M12 3.8a8.2 8.2 0 0 1 0 16.4z" fill="currentColor" stroke="none" />
    </svg>
  );
}
