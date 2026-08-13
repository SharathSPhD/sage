"""``strataq.evolutionary`` — populations that learn by copying.

:mod:`~strataq.evolutionary.replicator` has the deterministic large-population
dynamics: continuous and discrete replicator, rest points found exactly by
support enumeration and classified by tangent-space eigenvalues, and the logit
dynamic whose rest points are the logit QRE.
:mod:`~strataq.evolutionary.moran` has the finite-population stochastic side:
the birth–death chain under pairwise comparison, fixation probabilities against
the constant-selection closed form, and the stationary distribution.

The two modules meet at one scalar: the selection intensity β of the Fermi
imitation rule *is* the logit precision λ. :func:`compare_intensity` reports both
readings of the same game with the gap between them.
"""

from strataq.evolutionary.moran import (
    IntensityComparison,
    MoranChain,
    birth_death_stationary,
    compare_intensity,
    constant_selection_fixation,
    fermi,
    fixation_probability,
    moran_chain,
    pairwise_comparison_ratios,
    payoff_difference,
    small_mutation_stationary,
)
from strataq.evolutionary.replicator import (
    CENTRE,
    DEGENERATE,
    SADDLE,
    STABLE,
    UNSTABLE,
    RestPoint,
    discrete_replicator,
    logit_dynamic_field,
    logit_rest_point,
    replicator_field,
    replicator_flow,
    rest_points,
    stability,
    tangent_basis,
)

__all__ = [
    "CENTRE",
    "DEGENERATE",
    "SADDLE",
    "STABLE",
    "UNSTABLE",
    "IntensityComparison",
    "MoranChain",
    "RestPoint",
    "birth_death_stationary",
    "compare_intensity",
    "constant_selection_fixation",
    "discrete_replicator",
    "fermi",
    "fixation_probability",
    "logit_dynamic_field",
    "logit_rest_point",
    "moran_chain",
    "pairwise_comparison_ratios",
    "payoff_difference",
    "replicator_field",
    "replicator_flow",
    "rest_points",
    "small_mutation_stationary",
    "stability",
    "tangent_basis",
]
