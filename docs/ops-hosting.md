# SAGE — Hosting

**Decision: Oracle Cloud Always Free.** 2 OCPU / 12 GB ARM, one VM, €0/month.
Lives at `docs/ops/hosting.md`. For the Claude Code agent: §3 onward is executable in order.

---

## 1. What runs where

| | Where | Why |
|---|---|---|
| **Research compute** — α×λ sweeps, Hodge decompositions, EPR estimation, hierarchical fits | **DGX Spark, local** | Hours-long, no uptime requirement, results committed as static JSON |
| **Serving backend** — API, job queue, worker | **Oracle VM** | Always-on, bursty, small jobs only |
| **Database, auth, storage** | **Supabase free** | Keeps Postgres off the VM entirely |
| **Frontend** | **Vercel free** | Static + client fetch; heavy compute proxied to the API |
| **Demos, datasets, model weights** | **Hugging Face free** | Gradio Lab mirror, DreamPrice decoder |

**Never run the phase sweeps on the VM.** They belong in `experiments/`, run on the Spark, with results committed to `benchmarks/results/`. The dashboard reads files; it does not compute.

**Total cost: €0/month.**

---

## 2. The box

Oracle Always Free gives 1,500 OCPU-hours + 9,000 GB-hours per month. Divided by ~730 hours in a month that is exactly **2 OCPU + 12 GB running continuously** — not a burst budget to ration.

One OCPU is one full physical Ampere Neoverse N1 core (no SMT), so 2 OCPU is 2 real cores.

**Spec, for reference and for the fallback box:**

| | Floor | Target |
|---|---|---|
| RAM | 4 GB | **8 GB** (Oracle gives 12) |
| Cores | 2 physical | 2 dedicated / 4 shared |
| Disk | 40 GB | 80 GB (Oracle gives 200) |
| Swap | 1 GB | 2–4 GB |
| Arch | ARM64 | x86_64 is ~2–3× faster on float64 BLAS |

**Consequences of 2 ARM cores:** worker concurrency of 1, and solves take roughly 2–3× longer than on a modern EPYC. Fine for queued async jobs. Enforce the game-size limit (N=4, m=50, ~10⁷ states) in API validation *before* queueing.

**Fallback if Oracle ever fails you:** Netcup RS 1000 G12, ~€9.81/mo, 4 dedicated EPYC cores + 8 GB. Everything below is portable, so switching is one `docker compose` command against a different host.

---

## 3. Create the instance

Do this once. Shape must be exactly 2 OCPU / 12 GB to stay inside Always Free.

```bash
# Install and configure the OCI CLI (generates an API keypair locally)
bash -c "$(curl -L https://raw.githubusercontent.com/oracle/oci-cli/master/scripts/install/install.sh)"
oci setup config          # prompts for user OCID, tenancy OCID, region; say yes to key generation

cat ~/.oci/oci_api_key_public.pem
# Console → Profile → My profile → API keys → Add API key → paste the public key
# Confirm the fingerprint matches ~/.oci/config

chmod 600 ~/.oci/oci_api_key.pem ~/.oci/config
oci iam region list --output table     # verify
```

**Where the OCIDs live in the Console:** user OCID under Profile → My profile; tenancy OCID under Profile → Tenancy; compartment under Identity & Security → Compartments (on a personal account this is usually root, same as tenancy). Region identifier is top-right, e.g. `uk-london-1`.

```bash
export C=<compartment-ocid>

# Latest Ubuntu 24.04 for ARM
oci compute image list --compartment-id $C \
  --operating-system "Canonical Ubuntu" --operating-system-version "24.04" \
  --shape "VM.Standard.A1.Flex" --sort-by TIMECREATED --sort-order DESC --limit 1

oci compute instance launch \
  --compartment-id $C \
  --availability-domain "<AD-name>" \
  --shape "VM.Standard.A1.Flex" \
  --shape-config '{"ocpus": 2, "memoryInGBs": 12}' \
  --image-id "<image-ocid>" \
  --subnet-id "<subnet-ocid>" \
  --assign-public-ip true \
  --display-name "sage-api" \
  --boot-volume-size-in-gbs 100 \
  --metadata '{"ssh_authorized_keys": "<contents of ~/.ssh/id_ed25519.pub>"}'
```

Three things that trip everyone up:

- **One VM, not two.** The allowance can split across up to four instances. Don't — you want all 12 GB for the worker.
- **`Out of Capacity` is common.** Retry off-peak or try another availability domain. Script the retry.
- **Ports.** Open 80/443 in the subnet security list **and** on the host. Oracle's Ubuntu images ship restrictive `iptables` rules; this is the single most common "my server is unreachable" cause.

---

## 4. Prepare the server

```bash
ssh ubuntu@<public-ip>

# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu && newgrp docker

# Swap — JAX peaks are spiky; swap beats an OOM kill
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Host firewall (the Oracle gotcha)
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save

sudo mkdir -p /opt/sage && sudo chown ubuntu:ubuntu /opt/sage
```

---

## 5. The stack

`/opt/sage/docker-compose.yml`:

```yaml
services:
  caddy:
    image: caddy:2-alpine
    restart: unless-stopped
    ports: ["80:80", "443:443"]
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
    depends_on: [api]

  api:
    image: ghcr.io/sharathsphd/sage-api:latest
    restart: unless-stopped
    env_file: .env
    mem_limit: 4g
    command: uvicorn strataq_api.main:app --host 0.0.0.0 --port 8000
    depends_on: [redis]

  worker:
    image: ghcr.io/sharathsphd/sage-api:latest
    restart: unless-stopped
    env_file: .env
    mem_limit: 5g
    command: arq strataq_api.worker.WorkerSettings   # concurrency 1 — 2 cores only
    depends_on: [redis]

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    mem_limit: 256m
    command: redis-server --save "" --appendonly no

volumes:
  caddy_data:
```

`Caddyfile`:

```
api.yourdomain.com {
    reverse_proxy api:8000
}
```

`.env` (never committed): Supabase URL and service key, Redis URL, API key salt, `JAX_ENABLE_X64=1`, `JAX_PLATFORMS=cpu`.

Memory limits matter: they make a runaway job OOM its own container rather than the box.

---

## 6. Deploy

```
green gates → GitHub Actions builds linux/arm64 image → push to GHCR
           → SSH to instance → docker compose pull && up -d
```

**Build `linux/arm64`.** Ampere is aarch64; an amd64 image will not run. Set this in the workflow on day one:

```yaml
- uses: docker/build-push-action@v6
  with:
    platforms: linux/arm64
    push: true
    tags: ghcr.io/sharathsphd/sage-api:latest
```

Pin `jax[cpu]` in the image and warm the JIT on startup with a tiny game, so the first user request isn't a 20-second wait.

---

## 7. OCI MCP for Claude Code

**A correction if you were pointed at it:** Oracle's Console AI recommends the *Autonomous AI Database* MCP server. That gives an agent SQL access to an ADB instance — wrong tool. SAGE needs compute management, and its Postgres is in Supabase.

**Be honest about how little OCI automation this needs.** The VM is created once; everything after is SSH and Compose. So: **install OCI MCP read-only, and do mutations through the CLI behind confirmation.** Granting a fully-autonomous agent tenancy-wide write access buys little and creates real blast radius.

### 7.1 Least-privilege user

Console → Identity & Security. Create user `sage-agent`, group `sage-agents`, add the user, upload the API public key to *that* user, and point `~/.oci/config` at its OCID. Then policy `sage-agent-policy` in the root compartment:

```
Allow group sage-agents to read all-resources in tenancy
Allow group sage-agents to use instance-family in compartment <your-compartment>
Allow group sage-agents to read metrics in tenancy
```

That permits inspecting everything and start/stop/reboot — but not create, terminate, IAM, or networking. Widen deliberately if needed; never start from `manage all-resources`.

### 7.2 Install

Community OCI MCP servers read the standard `~/.oci/config`, so no secrets enter any JSON file. Best coverage is `jopsis/mcp-server-oci` (~95 tools, 11+ service categories, dynamic profile switching). Oracle also publishes first-party servers at `oracle.com/mcp` for Compute, Storage, Identity, Logging and Monitoring — check which are GA rather than preview.

```bash
cd ~/code/sage
claude mcp add oci --scope project -- uvx --from mcp-server-oci mcp-server-oci
claude mcp list
```

`.mcp.json` — safe to commit, contains no secrets:

```json
{
  "mcpServers": {
    "oci": {
      "command": "uvx",
      "args": ["--from", "mcp-server-oci", "mcp-server-oci"],
      "env": { "OCI_CLI_PROFILE": "DEFAULT", "OCI_REGION": "uk-london-1" }
    }
  }
}
```

If a server demands explicit env vars, reference the path (`"${HOME}/.oci/oci_api_key.pem"`), never the key contents.

### 7.3 Rails

In `.claude/settings.json`:

```json
{
  "permissions": {
    "deny": [
      "mcp__oci__terminate_*", "mcp__oci__delete_*", "mcp__oci__manage_iam_*",
      "Bash(oci * delete *)", "Bash(oci * terminate *)",
      "Read(~/.oci/**)", "Read(**/*.pem)"
    ],
    "ask": [
      "mcp__oci__create_*", "mcp__oci__update_*",
      "Bash(oci compute instance action *)", "Bash(ssh *)"
    ]
  }
}
```

`.gitignore`: `.oci/`, `*.pem`, `*.key`, `.env`. Keep the Stage 0 hook that blocks commits containing plausible credentials.

**Other MCPs worth having:** Supabase (already connected — genuinely useful for schema and data). GitHub needs no MCP; use `gh`.

---

## 8. Keep it alive

Free-tier instances under 10% CPU **and** 10% network over a rolling 7 days may be stopped. A low-traffic research app trips this reliably. Make the keep-alive do real work:

```cron
0 */4 * * * cd /opt/sage && docker compose run --rm api python -m strataq.bench.synthetic --publish
```

Spikes CPU, pushes results to Supabase (network traffic), and feeds the performance-trend chart on the Pages dashboard. One cron line, three jobs.

Also worth adding: a nightly `docker system prune -f`, and Uptime Kuma or a free external pinger on `/v1/health`.

---

## 9. Portability

Oracle halved this free tier once in 2026 with no announcement. The hedge is not avoidance, it's portability — and it's nearly free:

- Image in GHCR, not built on the box.
- Secrets in `.env`, not baked in.
- Data in Supabase, not on local disk.
- Artefacts regenerable via `make reproduce`.
- Deploy is one SSH target in a GitHub Actions secret.

If Oracle disappears, provision the Netcup box, change one secret, and redeploy. An afternoon and €10/month — but only if the above stays true from day one.

**Move off free when:** a user request routinely exceeds 30 s, you need concurrent heavy jobs, you're reclaimed or capacity-blocked twice, or you announce publicly.

---

## 10. Checklist

- [ ] `oci iam region list` works
- [ ] Instance is exactly 2 OCPU / 12 GB (`oci compute instance get`)
- [ ] 4 GB swap active (`free -h`)
- [ ] Ports 80/443 reachable — security list **and** host `iptables`
- [ ] arm64 image builds and runs
- [ ] `claude mcp list` shows `oci` connected
- [ ] Agent can report instance CPU utilisation
- [ ] Agent is *denied* when asked to terminate an instance
- [ ] `~/.oci/`, `*.pem`, `.env` gitignored and absent from `git log --all --stat`
- [ ] Idle-reclaim cron installed, output visible on the dashboard
- [ ] Health check green from outside the network
