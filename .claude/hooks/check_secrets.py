#!/usr/bin/env python3
"""Block commits containing plausible keys/secrets. Pre-commit mode (filenames as argv)."""

import re
import sys
from pathlib import Path

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("HF token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    (
        "generic assignment",
        re.compile(
            r"""(?i)\b(api_key|apikey|secret|token|password|passwd)\s*[:=]\s*['"][A-Za-z0-9_\-/+=]{16,}['"]"""
        ),
    ),
]

SKIP_SUFFIXES = {".lock", ".svg", ".png", ".ipynb"}


def main() -> int:
    findings = []
    for name in sys.argv[1:]:
        path = Path(name)
        if path.suffix in SKIP_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for label, pattern in PATTERNS:
            if pattern.search(text):
                findings.append(f"{name}: {label}")
    if findings:
        print(
            "BLOCKED: plausible secret in commit:\n  - "
            + "\n  - ".join(findings)
            + "\nSecrets come from the environment only, never the repo (master spec §18).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
