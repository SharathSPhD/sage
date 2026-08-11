#!/usr/bin/env python3
"""The plugin boundary: a domain is a plugin only if it never touches core.

Two checks, pre-commit mode (staged files as argv):
1. A commit that changes files under ``strataq/domains/`` AND files under
   ``strataq/core|finite|population/`` is blocked — a domain change that needs
   core changes means the "domain" is really an engine, which requires an ADR.
   Escape hatch: SAGE_ADR_REF=<adr-id> in the environment (used for deliberate,
   recorded engine work that happens to touch both).
2. Domain code may import ``strataq.core.*`` (the contract) and its OWN
   declared engine's modules (a plugin runs ON an engine — ADR-0008), but
   never the other engine and never another domain.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

DOMAIN_MARK = "strataq/domains/"
REPO = Path(__file__).resolve().parents[2]


def adr_ref_valid() -> bool:
    """An ADR reference is honoured only if the ADR actually exists (O-2).

    Sources, in order: the SAGE_ADR_REF env var (local commits), else the HEAD
    commit message (CI re-checks pushed commits, where the env var is gone but
    the reference is recorded in history — same strictness, auditable).
    """
    decisions = REPO / "memory" / "decisions.md"
    if not decisions.exists():
        return False
    known = decisions.read_text()
    ref = os.environ.get("SAGE_ADR_REF", "")
    if ref:
        return ref in known
    head = subprocess.run(
        ["git", "log", "-1", "--format=%B"], capture_output=True, text=True, check=False
    ).stdout
    return any(m in known for m in re.findall(r"ADR-\d{4}", head))


CORE_MARK = re.compile(r"strataq/(core|finite|population)/")
ENGINE_OR_DOMAIN_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+(strataq\.(?:finite|population|domains)[\w.]*)", re.MULTILINE
)
# accepts both the bare form and a typed one (`ENGINE: Literal[...] = "finite"`);
# a plugin that annotates its declaration must not silently lose its ADR-0008
# allowance — that false negative blocked a legitimate import on 2026-08-12
ENGINE_DECL = re.compile(
    r"^ENGINE(?:\s*:[^=]+)?\s*=\s*[\"\'](finite|population|bayesian)[\"\']", re.MULTILINE
)


def declared_engine(domain_file: str) -> str | None:
    """The ENGINE declared in the domain's __init__.py, if resolvable."""
    marker = domain_file.split(DOMAIN_MARK, 1)
    init = Path(marker[0] + DOMAIN_MARK + marker[1].split("/", 1)[0]) / "__init__.py"
    if not init.is_file():
        return None
    m = ENGINE_DECL.search(init.read_text())
    return m.group(1) if m else None


def main() -> int:
    files = sys.argv[1:]
    domain_files = [f for f in files if DOMAIN_MARK in f]
    core_files = [f for f in files if CORE_MARK.search(f)]

    if domain_files and core_files and not adr_ref_valid():
        print(
            "BLOCKED: this commit touches both domain plugins and core/engine code.\n"
            f"  domains: {', '.join(domain_files)}\n"
            f"  core:    {', '.join(core_files)}\n"
            "A domain that needs core changes is an engine; engines require an ADR "
            "(memory/decisions.md). If this IS recorded engine work, set "
            "SAGE_ADR_REF=<adr-id> and commit core and domain changes separately "
            "where possible.",
            file=sys.stderr,
        )
        return 1

    bad: list[str] = []
    for f in domain_files:
        if not f.endswith(".py") or not Path(f).is_file():
            continue  # deleted files in a commit have no content to scan
        own = f.split(DOMAIN_MARK, 1)[1].split("/", 1)[0]
        own_prefix = f"strataq.domains.{own}"
        engine = declared_engine(f)
        engine_prefix = f"strataq.{engine}" if engine else None
        for module in ENGINE_OR_DOMAIN_IMPORT.findall(Path(f).read_text()):
            # Own package: fine. Own declared engine: fine (ADR-0008).
            if module == own_prefix or module.startswith(own_prefix + "."):
                continue
            if engine_prefix and (
                module == engine_prefix or module.startswith(engine_prefix + ".")
            ):
                continue
            bad.append(f"{f} imports {module}")
            break
    if bad:
        print(
            "BLOCKED: domain plugins may import strataq.core.* (the contract) but "
            "never finite/population internals or other domains.\n"
            f"Files: {', '.join(bad)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
