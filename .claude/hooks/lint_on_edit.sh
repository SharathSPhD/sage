#!/usr/bin/env bash
# PostToolUse hook: auto-run ruff (and mypy for library files) on edited Python files.
# Receives Claude Code hook JSON on stdin; exit 2 feeds stderr back to Claude.
set -u
payload=$(cat)
file=$(printf '%s' "$payload" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' 2>/dev/null)
case "$file" in
  */sage/*.py) ;;
  *) exit 0 ;;
esac
cd "$(dirname "$0")/../.." || exit 0
out=$(uv run ruff check --fix "$file" 2>&1)
ruff_rc=$?
if [ $ruff_rc -ne 0 ]; then
  echo "ruff on $file:" >&2
  echo "$out" >&2
  exit 2
fi
case "$file" in
  *packages/strataq*/strataq*/*.py)
    out=$(uv run mypy "$file" 2>&1)
    if [ $? -ne 0 ]; then
      echo "mypy on $file:" >&2
      echo "$out" >&2
      exit 2
    fi
    ;;
esac
exit 0
