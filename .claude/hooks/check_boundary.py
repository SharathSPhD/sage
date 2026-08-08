#!/usr/bin/env python3
"""The plugin boundary: a domain is a plugin only if it never touches core.

Two checks, pre-commit mode (staged files as argv):
1. A commit that changes files under ``strataq/domains/`` AND files under
   ``strataq/core|finite|population/`` is blocked — a domain change that needs
   core changes means the "domain" is really an engine, which requires an ADR.
   Escape hatch: SAGE_ADR_REF=<adr-id> in the environment (used for deliberate,
   recorded engine work that happens to touch both).
2. Domain code may import ``strataq.core.*`` (it implements the contract) but
   never ``strataq.finite.*`` / ``strataq.population.*`` internals or another
   domain.
"""

import os
import re
import sys
from pathlib import Path

DOMAIN_MARK = "strataq/domains/"
REPO = Path(__file__).resolve().parents[2]


def adr_ref_valid() -> bool:
    """SAGE_ADR_REF is honoured only if the referenced ADR actually exists (O-2)."""
    ref = os.environ.get("SAGE_ADR_REF", "")
    if not ref:
        return False
    decisions = REPO / "memory" / "decisions.md"
    return decisions.exists() and ref in decisions.read_text()


CORE_MARK = re.compile(r"strataq/(core|finite|population)/")
ENGINE_OR_DOMAIN_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+(strataq\.(?:finite|population|domains)[\w.]*)", re.MULTILINE
)


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
        for module in ENGINE_OR_DOMAIN_IMPORT.findall(Path(f).read_text()):
            # A domain importing its own package is fine; anything else is not.
            if module == own_prefix or module.startswith(own_prefix + "."):
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
