"""strataq — stochastic strategic interaction in JAX.

The SAGE library. Facade: this module will expose ~15 public functions
(solve, decompose, response, dynamics, estimate); everything else is a
subpackage import. The facade fills in as Stage 1 units close their gates.

The import name is ``strataq``, never ``sage`` (SageMath owns that name).
"""

__version__ = "0.1.0.dev0"

# Float64 is a correctness requirement, not a preference: susceptibility and
# dissipation computations are ill-conditioned near criticality (PROGRAMME v3 §8.1).
from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

from strataq.core.protocols import (  # noqa: E402  (x64 config must precede any jax use)
    ActionGridBuilder,
    ConjugateFieldSpec,
    DatasetLoader,
    Engine,
    LearnPageSpec,
    PayoffOracle,
)

__all__ = [
    "ActionGridBuilder",
    "ConjugateFieldSpec",
    "DatasetLoader",
    "Engine",
    "LearnPageSpec",
    "PayoffOracle",
    "__version__",
]
