"""Smoke tests for the gate runner mechanics (not for any unit's actual state)."""

import importlib.util
import sys
from pathlib import Path

GATES = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_gates", GATES / "run_gates.py")
assert spec and spec.loader
run_gates = importlib.util.module_from_spec(spec)
sys.modules["run_gates"] = run_gates
spec.loader.exec_module(run_gates)


def test_stub_pattern_catches_stubs() -> None:
    assert run_gates.STUB_PATTERN.search("raise NotImplementedError")
    assert run_gates.STUB_PATTERN.search("# TODO: fix later")
    assert run_gates.STUB_PATTERN.search("pass  # implement later")
    assert not run_gates.STUB_PATTERN.search("def solve(): return x  # done")


def test_stage0_unit_is_listed_and_runs() -> None:
    assert "stage0" in run_gates.all_units()
    result = run_gates.run_unit("stage0")
    assert result["unit"] == "stage0"
    assert set(result["sections"]) == {
        "code",
        "domain",
        "statistical",
        "documentation",
        "adversarial",
    }


def test_conjectured_without_falsifier_is_red() -> None:
    gate = GATES / "_tmp_conjecture_test.yaml"
    gate.write_text("unit: _tmp_conjecture_test\nclaim: x\ntier: conjectured\ngates: {}\n")
    try:
        result = run_gates.run_unit("_tmp_conjecture_test")
        assert not result["green"]
        assert "falsifier" in str(result["sections"])
    finally:
        gate.unlink()
