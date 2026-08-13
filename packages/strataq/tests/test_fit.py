"""strataq.fit: the estimation workflow must recover lambda AND refuse to quote it.

Two claims are tested harder than the rest, because they are the reasons this
module exists:

* **recovery** -- on data simulated from a known lambda, ``fit`` returns it, and
  the interval covers it;
* **refusal** -- on data that cannot identify lambda, ``fit`` returns a BOUND and
  withholds the point estimate, rather than quoting a number the likelihood does
  not support.

Plus the thing aggregated counts cannot do: an interval clustered on subject.
"""

from __future__ import annotations

import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from strataq.core.solve.fixedpoint import logit_qre
from strataq.estimate.lam import sample_choices
from strataq.finite.games.library import rock_paper_scissors
from strataq.finite.games.tensor import DenseTensorGame
from strataq.fit import _chi2_sf, fit

# An asymmetric 3x3 game with a strict pure Nash at (0, 0): lambda is sharply
# identified here, and the lambda -> inf limit puts zero mass on observed choices.
ASYM = DenseTensorGame(
    (
        jnp.array([[3.0, 0.0, 1.5], [1.0, 2.0, 0.5], [0.0, 1.0, 2.5]]),
        jnp.array([[2.0, 1.0, 0.0], [0.5, 3.0, 1.0], [1.5, 0.0, 2.0]]),
    )
)

GRID = 80  # keep the suite quick; every fit re-uses one set of grid solves
BOOT = 200


def _rows(game, lam, n_per_player, seed, *, n_subjects=50, treatment=None):
    """Tidy long-form draws: one row per observed choice, panel structure intact."""
    sigma = [np.asarray(s, dtype=float) for s in logit_qre(game, lam).sigma]
    rng = np.random.default_rng(seed)
    rows = []
    for p, s in enumerate(sigma):
        draws = rng.choice(len(s), size=n_per_player, p=s / s.sum())
        for i, a in enumerate(draws):
            row = {
                "subject": f"p{p}s{i % n_subjects}",
                "player": p,
                "action": int(a),
                "round": i // n_subjects,
            }
            if treatment is not None:
                row["treatment"] = treatment
            rows.append(row)
    return rows


def _sticky_rows(game, lam, n_subjects, n_rounds, seed):
    """Maximal within-subject correlation: each subject repeats ONE action.

    The aggregate counts are unchanged in expectation; the effective sample size
    is n_subjects, not n_subjects * n_rounds. Any interval that ignores the panel
    will be far too narrow on this data, which is the point.
    """
    sigma = [np.asarray(s, dtype=float) for s in logit_qre(game, lam).sigma]
    rng = np.random.default_rng(seed)
    rows = []
    for p, s in enumerate(sigma):
        for i in range(n_subjects):
            a = int(rng.choice(len(s), p=s / s.sum()))
            for t in range(n_rounds):
                rows.append({"subject": f"p{p}s{i}", "player": p, "action": a, "round": t})
    return rows


def _columns(rows):
    return {k: [r[k] for r in rows] for k in rows[0]}


# ---------------------------------------------------------------------------
# recovery
# ---------------------------------------------------------------------------


def test_recovers_known_lambda_from_aggregated_counts():
    counts = sample_choices(ASYM, 1.2, 20_000, jax.random.key(11))
    f = fit(ASYM, counts, ci="profile", n_grid=GRID)
    assert f.identified
    assert abs(f.lam_hat - 1.2) / 1.2 < 0.15
    assert f.ci_low <= 1.2 <= f.ci_high
    assert f.n_obs == 40_000
    assert "profile likelihood" in f.ci_method


def test_recovers_known_lambda_from_a_tidy_table():
    rows = _rows(ASYM, 1.2, 8_000, seed=3)
    f = fit(ASYM, _columns(rows), ci="bootstrap", n_boot=BOOT, n_grid=GRID, seed=5)
    assert f.identified
    assert abs(f.lam_hat - 1.2) / 1.2 < 0.2
    assert f.ci_low <= 1.2 <= f.ci_high
    assert f.n_obs == 16_000 and f.n_subjects == 100


def test_tidy_and_counts_give_the_same_point_estimate():
    """The panel is kept for the UNCERTAINTY; the likelihood itself is identical."""
    rows = _rows(ASYM, 1.5, 4_000, seed=7)
    tidy = fit(ASYM, _columns(rows), ci="none", n_grid=GRID)
    counts = [
        np.bincount([r["action"] for r in rows if r["player"] == p], minlength=3) for p in range(2)
    ]
    agg = fit(ASYM, counts, ci="none", n_grid=GRID)
    assert tidy.lam_hat == pytest.approx(agg.lam_hat, rel=1e-12)
    assert tidy.loglik == pytest.approx(agg.loglik, rel=1e-12)


def test_list_of_records_and_mapping_are_both_accepted():
    rows = _rows(ASYM, 1.0, 1_000, seed=2)
    a = fit(ASYM, rows, ci="none", n_grid=GRID)
    b = fit(ASYM, _columns(rows), ci="none", n_grid=GRID)
    assert a.lam_hat == pytest.approx(b.lam_hat, rel=1e-12)


# ---------------------------------------------------------------------------
# the thing aggregated counts cannot do
# ---------------------------------------------------------------------------


def test_subject_clustering_widens_the_interval_that_counts_would_understate():
    rows = _sticky_rows(ASYM, 1.2, n_subjects=60, n_rounds=40, seed=13)
    clustered = fit(ASYM, _columns(rows), n_boot=BOOT, n_grid=GRID, seed=1)
    naive_rows = [{k: v for k, v in r.items() if k != "subject"} for r in rows]
    naive = fit(ASYM, _columns(naive_rows), n_boot=BOOT, n_grid=GRID, seed=1)

    assert "cluster bootstrap on subject" in clustered.ci_method
    assert clustered.n_subjects == 120
    assert "nonparametric bootstrap on choices" in naive.ci_method
    assert (clustered.ci_high - clustered.ci_low) > 1.5 * (naive.ci_high - naive.ci_low)
    assert any("no subject column" in w for w in naive.warnings)


def test_aggregated_counts_say_what_they_destroyed():
    counts = sample_choices(ASYM, 1.2, 2_000, jax.random.key(4))
    f = fit(ASYM, counts, n_boot=50, n_grid=GRID)
    assert any("aggregated counts supplied" in w for w in f.warnings)
    assert f.provenance["clustered_on"] == "none"


# ---------------------------------------------------------------------------
# refusal
# ---------------------------------------------------------------------------


def test_flat_likelihood_refuses_rather_than_quoting():
    """RPS is uniform at EVERY lambda: the likelihood is flat and lambda is not there."""
    rps = rock_paper_scissors()
    counts = [[13_400, 13_300, 13_300], [13_350, 13_350, 13_300]]
    f = fit(rps, counts, n_boot=50, n_grid=GRID)

    assert not f.identified
    assert f.kind == "unidentified"
    assert f.refusals and "NOT IDENTIFIED" in f.refusals[0]
    # the bound is the whole search window -- the widest TRUE statement
    assert f.ci_low == pytest.approx(0.05) and f.ci_high == pytest.approx(20.0)
    text = str(f.summary())
    assert "NOT IDENTIFIED" in text
    assert "lambda_hat         : NOT IDENTIFIED" in text
    assert "REFUSALS" in text
    # and it must NOT read like a point estimate anywhere in the headline
    assert f.as_dict()["lam_hat"] is None


def test_flat_likelihood_does_not_reject_uniform():
    rps = rock_paper_scissors()
    counts = [[13_400, 13_300, 13_300], [13_350, 13_350, 13_300]]
    f = fit(rps, counts, ci="none", n_grid=GRID)
    assert f.lr_uniform is not None
    assert f.lr_uniform.stat < 1.0
    assert f.lr_uniform.p > 0.1


# ---------------------------------------------------------------------------
# the two nested boundaries
# ---------------------------------------------------------------------------


def test_lr_against_uniform_rejects_on_informative_data():
    counts = sample_choices(ASYM, 1.2, 5_000, jax.random.key(9))
    f = fit(ASYM, counts, ci="none", n_grid=GRID)
    assert f.lr_uniform is not None
    assert f.lr_uniform.stat > 100.0
    assert f.lr_uniform.p < 1e-10
    assert f.lr_uniform.df == 1
    # the boundary correction is exactly half the chi2(1) tail, and is disclosed
    assert f.lr_uniform.p_boundary == pytest.approx(0.5 * f.lr_uniform.p)
    assert "boundary" in f.lr_uniform.note


def test_lr_against_nash_rejects_when_the_limit_zeroes_observed_choices():
    counts = sample_choices(ASYM, 1.2, 5_000, jax.random.key(21))
    f = fit(ASYM, counts, ci="none", n_grid=GRID)
    assert f.lr_nash is not None
    assert f.lr_nash.stat > 100.0
    assert f.lr_nash.p < 1e-10
    assert "lambda -> inf" in f.lr_nash.note or "lambda = " in f.lr_nash.note


def test_chi2_survival_matches_published_quantiles():
    assert _chi2_sf(3.841458820694124, 1) == pytest.approx(0.05, abs=1e-9)
    assert _chi2_sf(5.991464547107979, 2) == pytest.approx(0.05, abs=1e-9)
    assert _chi2_sf(7.814727903251179, 3) == pytest.approx(0.05, abs=1e-9)
    assert _chi2_sf(0.0, 1) == 1.0
    assert _chi2_sf(math.inf, 1) == 0.0


# ---------------------------------------------------------------------------
# by= : report the spread, never average it
# ---------------------------------------------------------------------------


def test_by_reports_per_group_lambda_and_flags_heterogeneity():
    rows = _rows(ASYM, 0.5, 4_000, seed=31, treatment="low") + _rows(
        ASYM, 3.0, 4_000, seed=32, treatment="high"
    )
    f = fit(ASYM, _columns(rows), by="treatment", n_boot=BOOT, n_grid=GRID, seed=2)

    assert len(f.groups) == 2
    keyed = {g.key: g for g in f.groups}
    assert abs(keyed["low"].lam_hat - 0.5) / 0.5 < 0.35
    assert abs(keyed["high"].lam_hat - 3.0) / 3.0 < 0.35
    assert f.homogeneity is not None
    assert f.homogeneity.df == 1
    assert f.homogeneity.p < 1e-3
    assert any("heterogeneous" in w for w in f.warnings)
    text = str(f.summary())
    assert "PER-GROUP LAMBDA" in text and "treatment" in text


def test_by_is_refused_on_aggregated_counts():
    counts = sample_choices(ASYM, 1.0, 500, jax.random.key(1))
    with pytest.raises(ValueError, match="by= needs a tidy table"):
        fit(ASYM, counts, by="treatment", n_grid=GRID)


# ---------------------------------------------------------------------------
# honesty affordances
# ---------------------------------------------------------------------------


def test_scale_fold_warning_is_always_present():
    counts = sample_choices(ASYM, 1.0, 800, jax.random.key(5))
    f = fit(ASYM, counts, ci="none", n_grid=GRID)
    assert any("not scale-free" in w.lower() or "scale fold" in w for w in f.warnings)
    assert f.provenance["lambda_normalised"] == pytest.approx(
        f.lam_hat * float(np.asarray(ASYM.payoff_range))
    )


def test_small_sample_is_warned_about():
    counts = sample_choices(ASYM, 1.0, 20, jax.random.key(6))
    f = fit(ASYM, counts, ci="none", n_grid=GRID)
    assert any("asymptotics" in w for w in f.warnings)


def test_summary_contains_every_promised_field():
    rows = _rows(ASYM, 1.2, 2_000, seed=41)
    f = fit(ASYM, _columns(rows), n_boot=BOOT, n_grid=GRID, seed=3)
    text = str(f.summary())
    for needle in (
        "lambda_hat",
        "95% CI",
        "cluster bootstrap on subject",
        "log-likelihood",
        "LIKELIHOOD-RATIO TESTS",
        "vs Nash (lambda -> inf)",
        "vs uniform (lambda = 0)",
        "df = 1",
        "WARNINGS",
        "PROVENANCE",
        "library_version",
    ):
        assert needle in text, needle
    assert repr(f.summary()) == text  # displays as itself in a REPL


def test_action_labels_are_mapped_in_sorted_order_and_warned_about():
    rows = _rows(rock_paper_scissors(), 1.0, 300, seed=8)
    labels = ["a_rock", "b_paper", "c_scissors"]
    for r in rows:
        r["action"] = labels[r["action"]]
    f = fit(rock_paper_scissors(), _columns(rows), ci="none", n_grid=GRID)
    assert f.provenance["action_label_order"] == labels
    assert any("sorted order" in w for w in f.warnings)


def test_symmetric_game_pools_roles_but_says_so():
    rows = _rows(rock_paper_scissors(), 1.0, 300, seed=12)
    stripped = [{k: v for k, v in r.items() if k != "player"} for r in rows]
    f = fit(rock_paper_scissors(), _columns(stripped), ci="none", n_grid=GRID)
    assert any("pooled onto player 0" in w for w in f.warnings)


def test_asymmetric_game_without_a_role_column_raises_instructively():
    rows = _rows(ASYM, 1.0, 200, seed=14)
    stripped = [{k: v for k, v in r.items() if k != "player"} for r in rows]
    with pytest.raises(ValueError, match="not symmetric"):
        fit(ASYM, _columns(stripped), n_grid=GRID)


# ---------------------------------------------------------------------------
# alternate routes and input guards
# ---------------------------------------------------------------------------


def test_agreement_method_reports_the_estimator_spread():
    counts = sample_choices(ASYM, 1.2, 3_000, jax.random.key(17))
    f = fit(ASYM, counts, method="agreement", ci="none", n_grid=GRID)
    assert f.agreement is not None
    assert set(f.agreement.estimates) >= {"mle", "mle_implicit", "dispersion"}
    assert "FOUR-ESTIMATOR AGREEMENT" in str(f.summary())


def test_bayes_method_gives_a_credible_interval():
    counts = sample_choices(ASYM, 1.2, 3_000, jax.random.key(19))
    f = fit(ASYM, counts, method="bayes", n_grid=GRID)
    assert "credible interval" in f.ci_method
    assert f.ci_low < f.lam_hat < f.ci_high
    assert abs(f.lam_hat - 1.2) / 1.2 < 0.25


def test_as_dict_is_json_serialisable():
    counts = sample_choices(ASYM, 1.2, 1_000, jax.random.key(23))
    f = fit(ASYM, counts, n_boot=50, n_grid=GRID)
    blob = json.dumps(f.as_dict())
    assert "lam_hat" in blob and "provenance" in blob


def test_input_guards():
    counts = sample_choices(ASYM, 1.0, 200, jax.random.key(2))
    with pytest.raises(TypeError, match="DenseTensorGame"):
        fit([[1.0, 2.0]], counts, n_grid=GRID)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one vector per player"):
        fit(ASYM, [counts[0]], n_grid=GRID)
    with pytest.raises(ValueError, match="must have length 3"):
        fit(ASYM, [[1, 2], [1, 2]], n_grid=GRID)
    with pytest.raises(ValueError, match="method must be"):
        fit(ASYM, counts, method="bogus", n_grid=GRID)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ci must be"):
        fit(ASYM, counts, ci="bogus", n_grid=GRID)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no action column"):
        fit(ASYM, {"player": [0, 1], "who": ["a", "b"]}, n_grid=GRID)
    with pytest.raises(ValueError, match="every count is zero"):
        fit(ASYM, [[0, 0, 0], [0, 0, 0]], n_grid=GRID)


def test_fit_is_reachable_from_the_facade():
    import strataq

    assert strataq.fit is fit
    assert "fit" in strataq.__all__


def test_plot_marks_lambda_and_overlays_the_observed_frequencies():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    counts = sample_choices(ASYM, 1.2, 2_000, jax.random.key(29))
    f = fit(ASYM, counts, ci="none", n_grid=GRID)
    ax = f.plot()
    labels = [t.get_text() for t in ax.texts]
    assert any("hat" in lab or "lambda" in lab for lab in labels)
    _handles, legend_labels = ax.get_legend_handles_labels()
    assert "observed frequency" in legend_labels
    assert ax.get_xscale() == "log"
