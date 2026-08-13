"""The Moran process in a finite population, and what β has to do with λ.

Two types in a population of ``N``. The state is the count ``i`` of type A, the
chain is birth–death, and everything follows from the ratio
``γ_i = T⁻(i) / T⁺(i)``: fixation probabilities are
``ρ_A = 1 / (1 + Σ_{k=1}^{N-1} Π_{j=1}^{k} γ_j)``, and — because a birth–death
chain is reversible — the stationary distribution of the mutation-perturbed chain
is the product ``π_i ∝ Π_{j=1}^{i} T⁺(j-1)/T⁻(j)``. No simulation anywhere.

**β is λ.** Under pairwise comparison a B-player copies an A-player with
probability ``1/(1 + e^{−β Δπ})``, and that is *identically* the logit choice
probability at precision λ = β over the two payoffs. So
``γ_i = e^{−β Δπ(i)}``, exponential-fitness Moran gives the same γ as Fermi
pairwise comparison, and the mean-field rest point of the same selection rule is
the symmetric logit QRE at λ = β. :func:`compare_intensity` computes both
readings of one game and reports the gap, so the claim is a number and not a
remark.

References
----------
Moran 1958; Nowak et al., Nature 2004 (fixation under weak selection);
Traulsen–Nowak–Pacheco, PRE 2006 (pairwise comparison and the Fermi rule);
Fudenberg–Imhof, JET 2006 (small-mutation limit); Blume 1993 (log-linear
response). Tier: exact for the birth–death algebra; the β = λ identification is
derived and tested here.
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from strataq.core.defaults import base_config
from strataq.core.solve.fixedpoint import logit_qre
from strataq.evolutionary.replicator import logit_rest_point
from strataq.finite.games.tensor import DenseTensorGame

__all__ = [
    "IntensityComparison",
    "MoranChain",
    "birth_death_stationary",
    "compare_intensity",
    "constant_selection_fixation",
    "fermi",
    "fixation_probability",
    "moran_chain",
    "pairwise_comparison_ratios",
    "payoff_difference",
    "small_mutation_stationary",
]


def fermi(payoff_difference_value: Array | float, beta: float) -> Array:
    """``1 / (1 + e^{−β Δπ})`` — the pairwise-comparison imitation probability.

    Identical to the two-action logit choice probability at precision λ = β; that
    identity is the whole content of "β is λ" and is tested to machine precision.
    """
    if float(beta) < 0:
        raise ValueError(f"beta must be >= 0, got {beta}")
    return jax.nn.sigmoid(float(beta) * jnp.asarray(payoff_difference_value, dtype=jnp.float64))


def _symmetric_matrix(payoff: Sequence[Sequence[float]] | Array) -> Array:
    a = jnp.asarray(payoff, dtype=jnp.float64)
    if a.shape != (2, 2):
        raise ValueError(
            f"the Moran chain here is two-type: need a 2x2 matrix, got {tuple(a.shape)}"
        )
    if not bool(jnp.all(jnp.isfinite(a))):
        raise ValueError("payoff matrix must be finite")
    return a


def payoff_difference(payoff: Sequence[Sequence[float]] | Array, population: int) -> Array:
    """``Δπ(i) = π_A(i) − π_B(i)`` for ``i = 0..N``, excluding self-interaction.

    ``π_A(i) = (a(i−1) + b(N−i)) / (N−1)`` and
    ``π_B(i) = (c i + d(N−i−1)) / (N−1)`` — the standard finite-population
    payoffs, where an individual does not play itself.
    """
    a = _symmetric_matrix(payoff)
    n = int(population)
    if n < 2:
        raise ValueError(f"population must be >= 2, got {n}")
    i = jnp.arange(n + 1, dtype=jnp.float64)
    pi_a = (a[0, 0] * (i - 1.0) + a[0, 1] * (n - i)) / (n - 1.0)
    pi_b = (a[1, 0] * i + a[1, 1] * (n - i - 1.0)) / (n - 1.0)
    return pi_a - pi_b


def pairwise_comparison_ratios(
    payoff: Sequence[Sequence[float]] | Array, population: int, beta: float
) -> Array:
    """``γ_i = T⁻(i)/T⁺(i) = e^{−β Δπ(i)}`` for ``i = 1..N−1``.

    Exponential-fitness Moran (``f = e^{βπ}``) yields the same ratios, so the two
    update rules cannot be told apart by fixation probabilities.
    """
    if float(beta) < 0:
        raise ValueError(f"beta must be >= 0, got {beta}")
    difference = payoff_difference(payoff, population)[1:-1]
    return jnp.exp(-float(beta) * difference)


def fixation_probability(ratios: Array) -> Array:
    """``ρ = 1 / (1 + Σ_{k=1}^{N-1} Π_{j=1}^{k} γ_j)`` from the ratio sequence.

    Computed through cumulative *log* sums, so a strongly selected chain with
    ``γ`` spanning many orders of magnitude does not overflow.
    """
    gamma = jnp.asarray(ratios, dtype=jnp.float64).ravel()
    if int(gamma.shape[0]) < 1:
        raise ValueError("need at least one ratio (population >= 2)")
    if bool(jnp.any(gamma <= 0.0)):
        raise ValueError("ratios must be strictly positive")
    log_products = jnp.cumsum(jnp.log(gamma))
    return 1.0 / (1.0 + jnp.sum(jnp.exp(log_products)))


def constant_selection_fixation(population: int, relative_fitness: float) -> Array:
    """``ρ = (1 − 1/r) / (1 − 1/r^N)`` — the closed form for constant selection.

    The ground truth every frequency-dependent fixation calculation must reduce
    to when ``Δπ`` does not depend on the state.

    References
    ----------
    Moran 1958; Nowak 2006 §6.3. Tier: exact.
    """
    n = int(population)
    r = float(relative_fitness)
    if n < 2:
        raise ValueError(f"population must be >= 2, got {n}")
    if r <= 0:
        raise ValueError(f"relative_fitness must be > 0, got {r}")
    if abs(r - 1.0) < 1e-12:
        return jnp.asarray(1.0 / n)
    return jnp.asarray((1.0 - 1.0 / r) / (1.0 - r ** (-n)))


class MoranChain(eqx.Module):
    """The birth–death chain over the count of type A, with mutation."""

    up: Array
    """``T⁺(i)``, the rate of ``i → i+1``, for ``i = 0..N``."""
    down: Array
    """``T⁻(i)``, the rate of ``i → i−1``, for ``i = 0..N``."""
    population: int = eqx.field(static=True)
    beta: Array
    mutation: Array

    @property
    def stationary(self) -> Array:
        """The stationary distribution over ``i = 0..N`` (reversible product form)."""
        return birth_death_stationary(self.up, self.down)

    @property
    def mean_share(self) -> Array:
        """``E[i/N]`` under the stationary distribution."""
        counts = jnp.arange(self.population + 1, dtype=jnp.float64) / self.population
        return self.stationary @ counts


def moran_chain(
    payoff: Sequence[Sequence[float]] | Array,
    population: int,
    beta: float,
    *,
    mutation: float | None = None,
) -> MoranChain:
    """Pairwise-comparison Moran chain with mutation.

    An individual is selected to revise; with probability ``μ`` it adopts a
    uniformly random type, otherwise it compares with a random other individual
    and imitates with the Fermi probability. ``μ > 0`` makes the chain ergodic —
    with ``μ = 0`` the monomorphic states absorb and no stationary distribution
    exists, which is why the mutation rate is a config value and not optional.
    """
    n = int(population)
    difference = payoff_difference(payoff, n)
    mu = float(base_config().evolutionary.mutation if mutation is None else mutation)
    if not 0.0 < mu < 1.0:
        raise ValueError(f"mutation must be in (0, 1), got {mu}")
    i = jnp.arange(n + 1, dtype=jnp.float64)
    share_a = i / n
    share_b = (n - i) / n
    imitate_up = fermi(difference, beta)
    up = share_b * ((1.0 - mu) * share_a * imitate_up + 0.5 * mu)
    down = share_a * ((1.0 - mu) * share_b * (1.0 - imitate_up) + 0.5 * mu)
    return MoranChain(
        up=up,
        down=down,
        population=n,
        beta=jnp.asarray(float(beta)),
        mutation=jnp.asarray(mu),
    )


def birth_death_stationary(up: Array, down: Array) -> Array:
    """``π_i ∝ Π_{j=1}^{i} T⁺(j−1)/T⁻(j)`` — reversibility, in logs."""
    plus = jnp.asarray(up, dtype=jnp.float64).ravel()
    minus = jnp.asarray(down, dtype=jnp.float64).ravel()
    if plus.shape != minus.shape:
        raise ValueError(f"up and down must match, got {plus.shape} and {minus.shape}")
    n = int(plus.shape[0]) - 1
    if n < 1:
        raise ValueError("need at least two states")
    ratios = jnp.log(plus[:n]) - jnp.log(minus[1:])
    log_weights = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(ratios)])
    return jnp.exp(jax.nn.log_softmax(log_weights))


def small_mutation_stationary(fixation_a: Array | float, fixation_b: Array | float) -> Array:
    """``(ρ_A, ρ_B) / (ρ_A + ρ_B)`` — time spent all-A vs all-B as ``μ → 0``.

    In the small-mutation limit the population is monomorphic almost always and
    the embedded chain jumps between the two absorbing states at rates
    proportional to the fixation probabilities.

    References
    ----------
    Fudenberg–Imhof, JET 2006. Tier: exact.
    """
    rho_a = jnp.asarray(fixation_a, dtype=jnp.float64)
    rho_b = jnp.asarray(fixation_b, dtype=jnp.float64)
    total = rho_a + rho_b
    return jnp.stack([rho_b / total, rho_a / total])


class IntensityComparison(eqx.Module):
    """One game read twice: as selection intensity β, and as logit precision λ.

    ``fermi_gap`` is the exact identity (imitation probability vs logit choice
    probability). ``qre_gap`` is the distance between the logit-dynamic rest
    point and the symmetric logit QRE from :mod:`strataq.finite` at λ = β; both
    are fixed points of ``x = softmax(β A x)``, so it is zero up to the solver
    tolerance. ``moran_share`` is what the finite population actually does at the
    same β.
    """

    beta: Array
    population: int = eqx.field(static=True)
    fermi_gap: Array
    """Max |imitation probability − logit choice probability| over the state space."""
    logit_rest_point: Array
    """``x`` solving ``x = softmax(β A x)``, ``(2,)``."""
    qre_symmetric: Array
    """Player 0's mix in the logit QRE of the same game at λ = β, ``(2,)``."""
    qre_gap: Array
    """Sup-norm distance between the two, ``(,)``."""
    moran_share: Array
    """``E[i/N]`` under the ergodic Moran stationary distribution."""
    moran_stationary: Array
    """The full stationary distribution over ``i = 0..N``."""
    fixation_a: Array
    fixation_b: Array
    monomorphic_weights: Array
    """Small-mutation weights on (all-B, all-A)."""

    @property
    def selected(self) -> int:
        """Which type the small-mutation limit favours: 0 for A, 1 for B."""
        return 0 if float(self.monomorphic_weights[1]) > float(self.monomorphic_weights[0]) else 1


def compare_intensity(
    payoff: Sequence[Sequence[float]] | Array,
    beta: float,
    population: int,
    *,
    mutation: float | None = None,
    tol: float | None = None,
) -> IntensityComparison:
    """Compute the evolutionary and the finite reading of one 2×2 game at β = λ.

    This is the comparison the project cares about, made a number: the same
    scalar controls the Fermi imitation rule in a finite population and the logit
    response in the strategic form, and the two readings of the same game are
    reported side by side with their gap.
    """
    a = _symmetric_matrix(payoff)
    n = int(population)
    chain = moran_chain(a, n, beta, mutation=mutation)
    difference = payoff_difference(a, n)
    # Exact identity: fermi(Δπ, β) is the two-action logit probability at λ = β.
    stacked = jnp.stack([difference, jnp.zeros_like(difference)], axis=-1)
    logit_choice = jnp.exp(jax.nn.log_softmax(float(beta) * stacked, axis=-1))[:, 0]
    fermi_gap = jnp.max(jnp.abs(fermi(difference, beta) - logit_choice))

    rest = logit_rest_point(a, beta, tol=tol)
    point = logit_qre(DenseTensorGame((a, a.T)), float(beta), tol=tol)
    qre = point.sigma[0]

    ratios = pairwise_comparison_ratios(a, n, beta)
    rho_a = fixation_probability(ratios)
    rho_b = rho_a * jnp.exp(jnp.sum(jnp.log(ratios)))
    return IntensityComparison(
        beta=jnp.asarray(float(beta)),
        population=n,
        fermi_gap=fermi_gap,
        logit_rest_point=rest,
        qre_symmetric=qre,
        qre_gap=jnp.max(jnp.abs(rest - qre)),
        moran_share=chain.mean_share,
        moran_stationary=chain.stationary,
        fixation_a=rho_a,
        fixation_b=rho_b,
        monomorphic_weights=small_mutation_stationary(rho_a, rho_b),
    )
