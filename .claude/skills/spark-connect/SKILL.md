---
name: spark-connect
description: Run work on the DGX Spark (spark-5208 / dgx-spark) from a Cowork cloud session. Use whenever a task must execute on the Spark — tests, gates, experiments, git, builds, deploys — or when SMB-mounted repo paths misbehave (rm refused, stale index.lock, phantom modified files). Covers the transport choice, the AppleScript escaping traps, the call-duration limit, and the long-job pattern.
---

# Working on the DGX Spark from a cloud session

## Topology — know which machine you are on

```
Cowork cloud container          the Mac mini                    the DGX Spark
(Bash tool runs HERE)           (osascript runs HERE)           (the real workspace)
   no route to the Spark  --->  ssh dgx-spark  ------------->   /home/sharaths/projects/sage
                                /Volumes/sharaths  <--SMB-----  /home/sharaths
                                     ^
                          device_bash sees this as $HOME/mnt/sage
```

Three facts that follow, and that cost real time to learn:

1. **The Spark is not reachable from the cloud container.** Its address is a Tailscale
   CGNAT IP (`100.98.74.5`, per the Mac's `~/.ssh/config`), so the container has no route.
   The Mac is a jump host, not a preference. If you want to remove the hop, the options are
   Tailscale on the container (needs an auth key — the user must do that, never handle it
   yourself) or a public SSH endpoint (security trade-off; ask first).
2. **`device_bash` runs on the Mac**, inside the Cowork VM, and only sees folders the user
   connected — `$HOME/mnt/sage` ↔ `/Volumes/sharaths/projects/sage` ↔ the Spark's
   `~/projects/sage`. **Sibling paths are invisible**: `~/projects/sage-wt/*` worktrees are
   NOT under the connected root, so `device_bash` cannot reach them.
3. **The `Bash` tool is the cloud container.** Nothing you run there touches the Spark.

## Transport: pick by what you are doing

| Task | Use |
|---|---|
| Run anything on the Spark | `osascript` → `ssh dgx-spark 'bash -s' <<'SCRIPT' … SCRIPT` |
| Author a script or patch file | **`device_bash` heredoc** into `$HOME/mnt/sage/_incoming/`, then run it over SSH |
| Move a file cloud → Spark | `SendUserFile` → `device_commit_files` to `/Volumes/sharaths/…` |
| Move a file Spark → cloud | write it under `~/projects/sage/`, then `device_stage_files` |
| Read a repo file | `device_stage_files` + `Read`, or `sed -n` over SSH |

## The escaping trap — this is the main hazard

`osascript` processes the AppleScript string **before** the shell sees it. Inside the
`do shell script "…"` payload:

- `\n` becomes a **real newline**. A Python patch containing `'…foo\n'` arrives split
  across lines as an unterminated string literal.
- `"` must be written `\"`, and one missed escape yields
  `syntax error: Expected " but found unknown token`.
- Nested heredocs with quotes inside (`python3 - <<'PY'` containing `"`) reliably break.

**Do not fight this.** Write the script with `device_bash` — an ordinary bash heredoc, no
mangling — into `$HOME/mnt/sage/_incoming/`, then invoke it with a *tiny* osascript line:

```
device_bash:  cat > $HOME/mnt/sage/_incoming/patch.py <<'PYEOF'
              …arbitrary Python, quotes and \n all safe…
              PYEOF

osascript:    ssh dgx-spark '~/projects/sage/.venv/bin/python ~/projects/sage/_incoming/patch.py <target>'
```

Keep osascript payloads short, single-quoted, and free of `\n` and `"`.

## Call-duration limit — never block

The osascript bridge caps a call at roughly two minutes. `sleep 100` inside a payload will
fail the whole call, and `with timeout of 900 seconds` does not raise the ceiling.

**Long jobs: start detached, poll in separate short calls.**

```
ssh dgx-spark 'nohup bash ~/projects/sage/_incoming/job.sh >/dev/null 2>&1 & echo STARTED'
ssh dgx-spark 'grep -E "DONE|FAIL" /tmp/job.log; tail -3 /tmp/job.log; true'
```

Have the job write sentinel markers (`STEP_OK`, `RUN_COMPLETE`, `SECONDS=…`) so a poll is
one cheap grep. `run_gates.py` and `pytest` buffer output, so poll `pgrep`/`ps -o etime=`
to distinguish "slow" from "hung".

## Gotchas that have actually bitten

- **`pkill -f <name>` matches substrings.** `pkill -f bench.sh` also kills `gate_bench.sh`.
  Name scripts distinctly and prefer an exact pattern.
- **`true` at the end of a payload.** `do shell script` raises on any non-zero exit, so a
  `grep` that finds nothing fails the call. End polls with `; true`.
- **`ssh` exit 255** is an SSH-layer error, not your command. Retry once before diagnosing.
- **The SMB mount cannot delete.** `rm` through `/Volumes/sharaths` returns
  `Operation not permitted`. Delete over SSH on the Spark instead.
- **Stale `.git/index.lock`** appears when a git command is interrupted through the mount,
  and blocks every later git op with "Another git process seems to be running". Check
  `pgrep -a git` first, then remove it over SSH.
- **Zero-byte "modified" files.** `git status` through the mount can show dozens of ` M`
  entries whose `git diff --stat` is `0` — a mount artifact, not real changes.
- **Never `git fetch` into a checked-out branch.** If a worktree has it, fetch to
  `FETCH_HEAD` and `git reset --hard FETCH_HEAD` inside that worktree.

## Environment on the Spark

- aarch64, 20 cores, NVIDIA GB10, ~1.3 TB free. jax **0.11** on **CPU** — the installed
  jaxlib is not CUDA-enabled, so JAX prints a GPU warning and falls back. Do not read that
  warning as an error.
- `uv` lives at `~/.local/bin/uv`; **export `PATH=$HOME/.local/bin:$PATH` in every payload**
  (each SSH call is a fresh non-login shell).
- `pdflatex` and `gh` are **absent**. Compile papers elsewhere; for GitHub state use the
  unauthenticated REST API (`api.github.com/repos/<owner>/<repo>/actions/runs`) — the repo
  is public, so no token is needed and none should be handled.
- Worktrees: `~/projects/sage` (main), `~/projects/sage-wt/{engine1,product,plane}`.
- `uv sync --all-packages` — plain `uv sync` prunes `strataq`.

## Deploy targets reachable *from the Spark*

- API VM: `ssh ubuntu@150.136.84.2` (passwordless sudo). Redeploy is
  `git pull && uv sync --package sage-api && sudo systemctl daemon-reload && sudo systemctl restart sage-api`,
  then check `http://150.136.84.2/v1/health`.
- Frontend: Vercel deploys from GitHub automatically — branch pushes become previews, `main`
  becomes production. Nothing to run by hand.

## Sanity check before starting

```
ssh dgx-spark 'echo OK; hostname; uname -m; export PATH=$HOME/.local/bin:$PATH; uv --version; cd ~/projects/sage && git rev-parse --abbrev-ref HEAD'
```
