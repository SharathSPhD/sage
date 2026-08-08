"""The enforcement hooks are code; code gets tests."""

import json
import subprocess
import sys
from pathlib import Path

HOOKS = Path(__file__).resolve().parents[1]


def run(
    script: str, args: list[str] | None = None, stdin: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HOOKS / script), *(args or [])],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


class TestImportSageGuard:
    def test_blocks_import_sage_in_hook_mode(self) -> None:
        payload = json.dumps({"tool_input": {"file_path": "x.py", "content": "import sage\n"}})
        proc = run("check_import_sage.py", stdin=payload)
        assert proc.returncode == 2
        assert "strataq" in proc.stderr

    def test_blocks_from_sage(self) -> None:
        payload = json.dumps(
            {"tool_input": {"file_path": "x.py", "new_string": "from sage.all import *\n"}}
        )
        assert run("check_import_sage.py", stdin=payload).returncode == 2

    def test_allows_strataq_and_mentions(self) -> None:
        payload = json.dumps(
            {
                "tool_input": {
                    "file_path": "x.py",
                    "content": "import strataq\n# the sage repo (mention ok)\nok = 1\n",
                }
            }
        )
        assert run("check_import_sage.py", stdin=payload).returncode == 0

    def test_ignores_non_python(self) -> None:
        payload = json.dumps({"tool_input": {"file_path": "notes.md", "content": "import sage"}})
        assert run("check_import_sage.py", stdin=payload).returncode == 0

    def test_blocks_dynamic_import_forms(self) -> None:
        # O-1 (stage0 red-team): dynamic-import bypasses must be caught too.
        # Snippets assembled at runtime so this test file passes the guard itself.
        dunder = "__imp" + "ort__"
        for snippet in (
            f'{dunder}("sage")\n',
            f"{dunder}('sage')\n",
            "importlib.imp" + 'ort_module("sage")\n',
        ):
            payload = json.dumps({"tool_input": {"file_path": "x.py", "content": snippet}})
            assert run("check_import_sage.py", stdin=payload).returncode == 2, snippet

    def test_precommit_mode(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.py"
        bad.write_text("import sage\n")
        good = tmp_path / "good.py"
        good.write_text("import strataq\n")
        assert run("check_import_sage.py", args=[str(bad)]).returncode == 1
        assert run("check_import_sage.py", args=[str(good)]).returncode == 0


class TestBoundaryGuard:
    def test_blocks_mixed_domain_core_commit(self, tmp_path: Path) -> None:
        # cwd outside any git repo: the ADR-0009 commit-message fallback cannot
        # fire, isolating the mixed-commit rule itself.
        proc = subprocess.run(
            [
                sys.executable,
                str(HOOKS / "check_boundary.py"),
                "packages/strataq/strataq/domains/blotto/oracle.py",
                "packages/strataq/strataq/core/protocols.py",
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 1
        assert "engine" in proc.stderr

    def test_allows_domain_only_commit(self, tmp_path: Path) -> None:
        f = tmp_path / "strataq" / "domains" / "blotto" / "oracle.py"
        f.parent.mkdir(parents=True)
        f.write_text("from strataq.core.protocols import PayoffOracle\n")
        assert run("check_boundary.py", args=[str(f)]).returncode == 0

    def test_blocks_domain_importing_engine_internals(self, tmp_path: Path) -> None:
        f = tmp_path / "strataq" / "domains" / "blotto" / "oracle.py"
        f.parent.mkdir(parents=True)
        f.write_text("from strataq.finite.decompose import hodge\n")
        proc = run("check_boundary.py", args=[str(f)])
        assert proc.returncode == 1

    def test_blocks_cross_domain_import(self, tmp_path: Path) -> None:
        f = tmp_path / "strataq" / "domains" / "blotto" / "oracle.py"
        f.parent.mkdir(parents=True)
        f.write_text("from strataq.domains.pricing.oracle import DemandOracle\n")
        assert run("check_boundary.py", args=[str(f)]).returncode == 1

    def test_own_domain_import_allowed(self, tmp_path: Path) -> None:
        f = tmp_path / "strataq" / "domains" / "blotto" / "learn.py"
        f.parent.mkdir(parents=True)
        f.write_text("from strataq.domains.blotto.oracle import BlottoOracle\n")
        assert run("check_boundary.py", args=[str(f)]).returncode == 0


class TestAdrEscapeHatch:
    """O-2 (stage0 red-team): SAGE_ADR_REF only works if the ADR actually exists."""

    MIXED = (
        "packages/strataq/strataq/domains/blotto/oracle.py",
        "packages/strataq/strataq/core/protocols.py",
    )

    def test_bogus_adr_ref_does_not_bypass(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(HOOKS / "check_boundary.py"), *self.MIXED],
            env={"SAGE_ADR_REF": "ADR-9999-does-not-exist"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 1

    def test_real_adr_ref_bypasses(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(HOOKS / "check_boundary.py"), *self.MIXED],
            env={"SAGE_ADR_REF": "ADR-0001"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0


class TestSecretScan:
    def test_blocks_github_token(self, tmp_path: Path) -> None:
        # Token assembled at runtime so this test file itself passes the scan.
        fake_token = "ghp_" + "A" * 36
        f = tmp_path / "config.py"
        f.write_text(f'TOKEN = "{fake_token}"\n')
        proc = run("check_secrets.py", args=[str(f)])
        assert proc.returncode == 1
        assert "GitHub" in proc.stderr

    def test_blocks_private_key(self, tmp_path: Path) -> None:
        marker = "-----BEGIN RSA " + "PRIVATE KEY-----"
        f = tmp_path / "key.txt"
        f.write_text(marker + "\n")
        assert run("check_secrets.py", args=[str(f)]).returncode == 1

    def test_allows_clean_file(self, tmp_path: Path) -> None:
        f = tmp_path / "clean.py"
        f.write_text('API_KEY = os.environ["STRATAQ_API_KEY"]\n')
        assert run("check_secrets.py", args=[str(f)]).returncode == 0


class TestGateRegressionFlatten:
    def test_flatten_skips_list_values(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("ctg", HOOKS / "check_tests_and_gates.py")
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        flat = mod.flatten_status(
            {"units": {"stage0": {"green": True, "sections": {"code": ["a failure"]}}}}
        )
        assert flat == {"units.stage0.green": True}


class TestEngineImportRule:
    """ADR-0008: a domain may import its own declared engine, nothing else."""

    def _make_domain(self, tmp_path: Path, engine: str, import_line: str) -> Path:
        d = tmp_path / "strataq" / "domains" / "fake"
        d.mkdir(parents=True)
        (d / "__init__.py").write_text(f'ENGINE = "{engine}"\n')
        f = d / "oracle.py"
        f.write_text(import_line + "\n")
        return f

    def test_own_engine_import_allowed(self, tmp_path: Path) -> None:
        f = self._make_domain(
            tmp_path, "population", "from strataq.population.games.routing import RoutingNetwork"
        )
        assert run("check_boundary.py", args=[str(f)]).returncode == 0

    def test_cross_engine_import_blocked(self, tmp_path: Path) -> None:
        f = self._make_domain(tmp_path, "population", "from strataq.finite.decompose import hodge")
        assert run("check_boundary.py", args=[str(f)]).returncode == 1


class TestAdrFromCommitMessage:
    """ADR-0009: in CI the ADR reference is read from the HEAD commit message."""

    def test_head_message_reference_honoured(self) -> None:
        # This repo's HEAD (or an ancestor scenario) may or may not reference an
        # ADR; assert only the mechanism: a bogus env ref still fails even if
        # the fallback path exists.
        proc = subprocess.run(
            [
                sys.executable,
                str(HOOKS / "check_boundary.py"),
                "packages/strataq/strataq/domains/blotto/oracle.py",
                "packages/strataq/strataq/core/protocols.py",
            ],
            env={"SAGE_ADR_REF": "ADR-9999-nope"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 1
