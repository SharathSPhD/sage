"""The naming rule, as an executable fact.

``import sage`` must never work from this codebase's own packages: the
distribution is ``strataq``. This test asserts the workspace does not provide a
``sage`` module (SageMath, if a user installs it, is their business — but WE
never ship or shadow one), and that float64 is on by default.
"""

import importlib.util


def test_workspace_provides_no_sage_module() -> None:
    spec = importlib.util.find_spec("sage")
    assert spec is None, (
        "A 'sage' module is importable from this environment. The library is "
        "'strataq'; nothing in this workspace may create or shadow 'sage'."
    )


def test_x64_enabled_on_import() -> None:
    import jax.numpy as jnp
    import strataq  # noqa: F401  (import triggers the config)

    assert jnp.asarray(1.0).dtype == jnp.float64
