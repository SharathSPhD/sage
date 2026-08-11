"""Release-integrity tests (caught two real defects at 0.1.0 publish time).

The 0.1.0 smoke test found that an INSTALLED strataq could not run any
solver call — `config/base.yaml` lives in the repo, and the packaged
fallback that `base_config()` documents was never actually shipped. These
tests make both failure modes permanent.
"""

from pathlib import Path

import strataq
from strataq.core.defaults import base_config

PKG = Path(strataq.__file__).parent
REPO_CONFIG = PKG.parents[2] / "config" / "base.yaml"


def test_packaged_base_config_exists():
    """The installed-package fallback must ship inside the wheel."""
    assert (PKG / "core" / "base.yaml").is_file()


def test_packaged_config_matches_repo_config():
    """Drift guard: the packaged copy must be byte-identical to the repo's."""
    if not REPO_CONFIG.is_file():  # installed (non-checkout) environment
        return
    assert (PKG / "core" / "base.yaml").read_bytes() == REPO_CONFIG.read_bytes()


def test_packaged_config_loads_standalone():
    """The fallback alone must satisfy the validated schema."""
    cfg = base_config(str(PKG / "core" / "base.yaml"))
    assert cfg.tolerances.identity > 0
    assert cfg.numerics.dtype == "float64"


def test_version_matches_pyproject():
    """__version__ must track the distribution metadata, not a hand-edited
    literal — and in a checkout it must equal pyproject's version (a stale
    editable install is itself the drift this catches)."""
    pyproject = PKG.parents[1] / "pyproject.toml"
    if not pyproject.is_file():  # installed (non-checkout) environment
        assert strataq.__version__
        return
    declared = next(
        line.split("=", 1)[1].strip().strip('"')
        for line in pyproject.read_text().splitlines()
        if line.startswith("version =")
    )
    assert strataq.__version__ == declared, (
        f"{strataq.__version__} != {declared}: re-sync the editable install"
    )
