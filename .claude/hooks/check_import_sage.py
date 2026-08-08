#!/usr/bin/env python3
"""Block `import sage` / `from sage ...` — the library is `strataq`.

Two modes:
- pre-commit: filenames as argv; scan file contents.
- Claude Code PreToolUse: JSON on stdin (Edit/Write tool input); scan the
  content being written. Exit 2 blocks the tool call with the stderr message.
"""

import json
import re
import sys
from pathlib import Path

# Literal imports, and the dynamic forms worth catching (O-1, stage0 red-team).
# String-eval smuggling (exec of assembled strings) is unbounded and out of
# scope for a regex guard; the backstop is test_workspace_provides_no_sage_module —
# the workspace ships no `sage` module, so any such import fails at runtime.
PATTERN = re.compile(
    r"^\s*(import\s+sage\b|from\s+sage\b)"
    r"|__import__\(\s*['\"]sage['\"]"
    r"|import_module\(\s*['\"]sage['\"]",
    re.MULTILINE,
)
MESSAGE = (
    "BLOCKED: `import sage` (or a dynamic-import equivalent) detected. The library "
    "is `strataq` — SageMath owns the `sage` import (root CLAUDE.md, naming rule)."
)


def scan_text(text: str) -> bool:
    return bool(PATTERN.search(text))


def main() -> int:
    if len(sys.argv) > 1:  # pre-commit mode
        bad = [f for f in sys.argv[1:] if f.endswith(".py") and scan_text(Path(f).read_text())]
        if bad:
            print(f"{MESSAGE}\nFiles: {', '.join(bad)}", file=sys.stderr)
            return 1
        return 0
    # hook mode: PreToolUse JSON on stdin
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    tool_input = payload.get("tool_input", {})
    path = str(tool_input.get("file_path", ""))
    if not path.endswith(".py"):
        return 0
    content = "".join(str(tool_input.get(k, "")) for k in ("content", "new_string", "new_source"))
    if scan_text(content):
        print(MESSAGE, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
