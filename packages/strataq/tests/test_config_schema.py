"""config/base.yaml must round-trip through the typed schema — the config contract."""

from pathlib import Path

import yaml
from strataq.core.config import BaseConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_base_yaml_validates() -> None:
    raw = yaml.safe_load((REPO_ROOT / "config" / "base.yaml").read_text())
    cfg = BaseConfig.from_mapping(raw)
    assert cfg.numerics.jax_enable_x64 is True
    assert cfg.tolerances.identity <= 1e-12
    assert cfg.tolerances.fd <= 1e-6
    assert cfg.criticality.warn_below <= 1e-3


def test_tolerances_are_ordered_sanely() -> None:
    raw = yaml.safe_load((REPO_ROOT / "config" / "base.yaml").read_text())
    tol = BaseConfig.from_mapping(raw).tolerances
    # identity is the strictest rung; finite differences the loosest.
    assert tol.identity <= tol.decompose <= tol.oracle <= tol.fd
