"""strataq — stochastic strategic interaction in JAX.

The SAGE library. Facade: this module will expose ~15 public functions
(solve, decompose, response, dynamics, estimate); everything else is a
subpackage import. The facade fills in as Stage 1 units close their gates.

The import name is ``strataq``, never ``sage`` (SageMath owns that name).
"""

try:  # single source of truth: the installed distribution metadata
    from importlib.metadata import version as _version

    __version__ = _version("strataq")
except Exception:  # editable/source checkout without metadata
    __version__ = "0.1.0"

# Float64 is a correctness requirement, not a preference: susceptibility and
# dissipation computations are ill-conditioned near criticality (PROGRAMME v3 §8.1).
from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

from strataq import games  # noqa: E402  (x64 config must precede any jax use)
from strataq.core.protocols import (  # noqa: E402
    ActionGridBuilder,
    ConjugateFieldSpec,
    DatasetLoader,
    Engine,
    LearnPageSpec,
    PayoffOracle,
)
from strataq.core.solve.fixedpoint import logit_qre  # noqa: E402
from strataq.core.types import Game, QREPoint, SpectrumInfo  # noqa: E402
from strataq.diagnose import Diagnosis, diagnose  # noqa: E402
from strataq.finite.decompose.generate import make_family  # noqa: E402
from strataq.finite.decompose.hodge import alpha, hodge_decompose  # noqa: E402
from strataq.finite.games.normalise import normalise  # noqa: E402
from strataq.finite.games.tensor import DenseTensorGame  # noqa: E402
from strataq.finite.response.reciprocity import reciprocity_defect  # noqa: E402
from strataq.finite.response.spectral import critical_lambda, strategic_spectrum  # noqa: E402
from strataq.finite.response.susceptibility import chi_equilibrium, chi_partial  # noqa: E402
from strataq.fit import LambdaFit, fit  # noqa: E402

__all__ = [
    "ActionGridBuilder",
    "ConjugateFieldSpec",
    "DatasetLoader",
    "DenseTensorGame",
    "Diagnosis",
    "Engine",
    "Game",
    "LambdaFit",
    "LearnPageSpec",
    "PayoffOracle",
    "QREPoint",
    "SpectrumInfo",
    "__version__",
    "alpha",
    "chi_equilibrium",
    "chi_partial",
    "critical_lambda",
    "diagnose",
    "fit",
    "games",
    "hodge_decompose",
    "logit_qre",
    "make_family",
    "normalise",
    "reciprocity_defect",
    "strategic_spectrum",
]
