"""Contract tests for the plugin protocols — the one real Stage 0 code unit."""

from dataclasses import FrozenInstanceError

import pytest
from strataq.core.protocols import (
    ActionGridBuilder,
    ConjugateFieldSpec,
    DatasetLoader,
    DomainPlugin,
    LearnPageSpec,
    PayoffOracle,
)


class _FakeOracle:
    n_players = 2

    def profit(self, actions, state=None):
        return actions

    def quantity(self, actions, state=None):
        return actions

    def response_matrix(self, actions, state=None):
        return actions


class _FakeGrid:
    def build(self):
        return ()


class _FakeLoader:
    def load(self):
        return None

    def validate(self):
        return "ok"


def test_structural_conformance() -> None:
    assert isinstance(_FakeOracle(), PayoffOracle)
    assert isinstance(_FakeGrid(), ActionGridBuilder)
    assert isinstance(_FakeLoader(), DatasetLoader)


def test_none_sentinel_is_unobservable_and_frozen() -> None:
    none = ConjugateFieldSpec.NONE
    assert none.observable is False
    with pytest.raises(FrozenInstanceError):
        none.observable = True  # type: ignore[misc]


def test_observable_field_requires_name() -> None:
    with pytest.raises(ValueError):
        ConjugateFieldSpec(name="")
    tolls = ConjugateFieldSpec(name="link tolls", linearity="exact")
    assert tolls.observable


def test_plugin_instrument_availability_keys_off_field() -> None:
    learn = LearnPageSpec(slug="blotto", title="Colonel Blotto")
    with_field = DomainPlugin(
        name="blotto",
        engine="finite",
        oracle_factory=_FakeOracle,
        grid_factory=_FakeGrid,
        field_spec=ConjugateFieldSpec(name="battlefield budget"),
        loader_factory=None,
        learn=learn,
    )
    without_field = DomainPlugin(
        name="sports",
        engine="finite",
        oracle_factory=_FakeOracle,
        grid_factory=_FakeGrid,
        field_spec=ConjugateFieldSpec.NONE,
        loader_factory=None,
        learn=LearnPageSpec(slug="sports", title="Penalty kicks"),
    )
    assert with_field.response_instruments_available
    assert not without_field.response_instruments_available
