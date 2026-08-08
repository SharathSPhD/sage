"""Library defaults, loaded once from config/base.yaml — never inline literals.

Library functions take ``tol=None`` / ``max_iter=None`` and resolve through
:func:`base_config`. Experiments override via Hydra composition; the library
itself only ever reads the validated schema.

References
----------
Master spec §7 (config-driven). Engineering invariant.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from strataq.core.config import BaseConfig

_REPO_CONFIG = Path(__file__).resolve().parents[4] / "config" / "base.yaml"


@lru_cache(maxsize=1)
def base_config(path: str | None = None) -> BaseConfig:
    """The validated base configuration (cached).

    Resolution order: explicit ``path`` argument, else the repo's
    ``config/base.yaml`` (present in development checkouts), else the packaged
    fallback next to this module.
    """
    candidates = [Path(path)] if path else [_REPO_CONFIG, Path(__file__).parent / "base.yaml"]
    for candidate in candidates:
        if candidate.exists():
            raw = yaml.safe_load(candidate.read_text())
            return BaseConfig.from_mapping(raw)
    raise FileNotFoundError(f"No base config found (searched {candidates}).")
