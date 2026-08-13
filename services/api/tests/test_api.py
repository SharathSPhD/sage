"""API surface tests: every endpoint against calibration-known games."""

import jax.numpy as jnp
from fastapi.testclient import TestClient
from sage_api.main import app

client = TestClient(app)

RPS = {
    "payoffs": [
        [[0, -1, 1], [1, 0, -1], [-1, 1, 0]],
        [[0, 1, -1], [-1, 0, 1], [1, -1, 0]],
    ],
    "lam": 1.5,
}
COORD = {"payoffs": [[[2, 0], [0, 2]], [[2, 0], [0, 2]]], "lam": 1.0}


def test_health():
    body = client.get("/v1/health").json()
    assert body["status"] == "ok"


def test_solve_qre_uniform_on_rps():
    body = client.post("/v1/solve/qre", json=RPS).json()
    for s in body["sigma"]:
        assert all(abs(p - 1 / 3) < 1e-8 for p in s)
    assert body["provenance"]["payoff_range"] == 2.0
    assert body["provenance"]["lambda_normalised"] == 3.0


def test_decompose_reads_alpha_one_on_rps_and_zero_on_coordination():
    assert client.post("/v1/decompose", json=RPS).json()["alpha"] > 1 - 1e-9
    assert client.post("/v1/decompose", json=COORD).json()["alpha"] < 1e-9


def test_response_reads_R_zero_on_coordination_positive_on_rps():
    coord = client.post("/v1/response", json=COORD).json()
    rps = client.post("/v1/response", json=RPS).json()
    assert coord["reciprocity_defect"] < 1e-10
    assert rps["reciprocity_defect"] > 0.1


def test_supercritical_warning_fires():
    body = client.post("/v1/response", json={**RPS, "lam": 10.0}).json()
    assert any("supercritical" in w or "near_criticality" in w for w in body["warnings"])


def test_dynamics_detailed_balance_split():
    coord = client.post("/v1/dynamics/stationary", json=COORD).json()
    rps = client.post("/v1/dynamics/stationary", json={**RPS, "lam": 2.0}).json()
    assert coord["detailed_balance"] and coord["epr"] < 1e-12
    assert not rps["detailed_balance"] and rps["epr"] > 1e-3


def test_branch_traces():
    body = client.post(
        "/v1/solve/branch", json={"payoffs": COORD["payoffs"], "lam_max": 3.0, "n_points": 150}
    ).json()
    assert body["lambdas"][-1] >= 3.0


def test_size_guard():
    big = {"payoffs": [[[0.0] * 20 for _ in range(20)]] * 2, "lam": 1.0}
    assert client.post("/v1/solve/qre", json=big).status_code == 413


def test_invalid_game_rejected():
    bad = {"payoffs": [[[1, 2], [3, 4]], [[1, 2, 3], [4, 5, 6]]], "lam": 1.0}
    assert client.post("/v1/solve/qre", json=bad).status_code == 422


def test_examples_cover_both_anchors():
    body = client.get("/v1/examples").json()
    assert any("alpha=0" in k for k in body) and any("alpha=1" in k for k in body)


def test_empty_payload_rejected_not_crashed():
    """Red-team finding: empty payoffs must 422, never 500."""
    resp = client.post("/v1/solve/qre", json={"payoffs": [], "lam": 1.0})
    assert resp.status_code == 422


def test_nonfinite_payoffs_rejected():
    """Python's json parser accepts literal Infinity/NaN; the guard must catch it."""
    raw = '{"payoffs": [[[Infinity, 0], [0, 1]], [[1, 0], [0, 1]]], "lam": 1.0}'
    resp = client.post("/v1/solve/qre", content=raw, headers={"content-type": "application/json"})
    assert resp.status_code in (400, 422)


def test_poke_cross_readings_symmetric_on_potential():
    """Doc 07's measurement procedure: poke 1 read 2 vs poke 2 read 1."""
    base = {"payoffs": COORD["payoffs"], "lam": 1.0, "player": 0, "action": 0, "size": 0.2}
    r1 = client.post("/v1/response/poke", json=base).json()
    r2 = client.post("/v1/response/poke", json={**base, "player": 1}).json()
    assert r1["delta"][0][0] > 0  # own action pushed up
    # cross-reading: how much the OTHER player moved
    cross_12 = r1["delta"][1][0]
    cross_21 = r2["delta"][0][0]
    assert abs(cross_12 - cross_21) < 1e-6  # reciprocity on an exact potential game
    assert "sigma_base" in r1 and "sigma_poked" in r1


def test_poke_rejects_bad_indices():
    bad = {"payoffs": COORD["payoffs"], "lam": 1.0, "player": 5, "action": 0, "size": 0.1}
    assert client.post("/v1/response/poke", json=bad).status_code == 422


def test_stationary_includes_currents_and_states():
    rps = client.post("/v1/dynamics/stationary", json=RPS).json()
    coord = client.post("/v1/dynamics/stationary", json=COORD).json()
    n = len(rps["pi"])
    assert len(rps["currents"]) == n and len(rps["currents"][0]) == n
    assert len(rps["states"]) == n and len(rps["states"][0]) == 2
    flat = [abs(v) for row in rps["currents"] for v in row]
    assert max(flat) > 1e-4  # RPS circulates
    flat_c = [abs(v) for row in coord["currents"] for v in row]
    assert max(flat_c) < 1e-10  # potential game: no current


def test_sample_estimates_track_exact():
    body = {**RPS, "n_steps": 8000, "n_trajectories": 8, "seed": 7}
    out = client.post("/v1/dynamics/sample", json=body).json()
    assert out["exact_epr"] > 1e-3
    assert abs(out["kld_epr"] - out["exact_epr"]) / out["exact_epr"] < 0.5
    assert 0 < out["tur_ci_low"] <= out["exact_epr"] * 1.1
    again = client.post("/v1/dynamics/sample", json=body).json()
    assert again["kld_epr"] == out["kld_epr"]  # seeded, deterministic


def test_sample_size_guard():
    body = {**RPS, "n_steps": 500000, "n_trajectories": 64, "seed": 1}
    assert client.post("/v1/dynamics/sample", json=body).status_code in (413, 422)


def test_estimate_lambda_recovers_and_warns():
    import jax
    import jax.numpy as jnp
    from strataq.estimate.lam import sample_choices
    from strataq.finite.games.tensor import DenseTensorGame

    game = DenseTensorGame(
        (
            jnp.array([[3.0, 0.0, 1.5], [1.0, 2.0, 0.5], [0.0, 1.0, 2.5]]),
            jnp.array([[2.0, 1.0, 0.0], [0.5, 3.0, 1.0], [1.5, 0.0, 2.0]]),
        )
    )
    counts = sample_choices(game, 1.2, 20_000, jax.random.key(11))
    body = {
        "payoffs": [u.tolist() for u in game.payoffs],
        "counts": [[int(x) for x in c] for c in counts],
    }
    out = client.post("/v1/estimate/lambda", json=body).json()
    assert abs(out["estimates"]["mle"]["lam"] - 1.2) / 1.2 < 0.15
    assert out["agreement_gap"] < 0.2

    rps_counts = [[13400, 13300, 13300], [13350, 13350, 13300]]
    rps_body = {"payoffs": RPS["payoffs"], "counts": rps_counts}
    out2 = client.post("/v1/estimate/lambda", json=rps_body).json()
    assert any("unidentified" in w for w in out2["warnings"])


def test_sioux_network_and_sue():
    net = client.get("/v1/domains/sioux_falls/network").json()
    assert net["n_nodes"] == 24 and len(net["links"]) == 76
    base = client.post("/v1/domains/sioux_falls/sue", json={"theta": 0.5}).json()
    assert base["residual"] < 1e-8
    assert base["total_travel_time"] > 0
    # toll the busiest link: flow there must FALL, total time must not improve
    busiest = max(range(76), key=lambda i: base["link_flows"][i])
    tolled = client.post(
        "/v1/domains/sioux_falls/sue", json={"theta": 0.5, "tolls": {str(busiest): 20.0}}
    ).json()
    assert tolled["link_flows"][busiest] < base["link_flows"][busiest]


def test_sioux_sue_guards():
    assert (
        client.post(
            "/v1/domains/sioux_falls/sue", json={"theta": 0.5, "tolls": {"999": 1.0}}
        ).status_code
        == 422
    )


def test_toolkit_reciprocity_f0011():
    r = client.post(
        "/v1/toolkit/reciprocity",
        json={"chi": [[1.0697, 0.0028], [0.0005, 0.9685]]},
    )
    assert r.status_code == 200
    body = r.json()
    assert abs(body["r"] - 0.0011) < 3e-4
    assert body["warnings"]  # honesty warnings travel over HTTP too


def test_toolkit_reciprocity_with_se_gives_ci():
    r = client.post(
        "/v1/toolkit/reciprocity",
        json={"chi": [[1.0, 0.05], [0.01, 1.0]], "chi_se": [[0.02, 0.02], [0.02, 0.02]]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ci_low"] is not None and body["ci_low"] < body["r"] < body["ci_high"]


def test_toolkit_reciprocity_nan_is_422():
    r = client.post("/v1/toolkit/reciprocity", json={"chi": [[1.0, None], [0.2, 0.9]]})
    assert r.status_code == 422


def test_toolkit_irreversibility_walk_at_null():
    import numpy as np

    series = list(np.cumsum(np.random.default_rng(0).normal(size=400)))
    r = client.post(
        "/v1/toolkit/irreversibility",
        json={"series": series, "n_surrogates": 60},
    )
    assert r.status_code == 200
    assert r.json()["detected"] is False


def test_toolkit_irreversibility_constant_is_422():
    r = client.post("/v1/toolkit/irreversibility", json={"series": [5.0] * 200})
    assert r.status_code == 422
    assert "constant" in r.json()["detail"]


def test_toolkit_rationality_flat_warns():
    r = client.post(
        "/v1/toolkit/rationality",
        json={
            "payoff_matrices": [[[1.0, -1.0], [-1.0, 1.0]], [[-1.0, 1.0], [1.0, -1.0]]],
            "counts": [[500, 500], [500, 500]],
        },
    )
    assert r.status_code == 200
    assert any("flat likelihood" in w for w in r.json()["warnings"])


def test_blotto_read_symmetric():
    r = client.post("/v1/domains/blotto/read", json={"budget_a": 3, "budget_b": 3})
    assert r.status_code == 200
    body = r.json()
    assert abs(body["alpha"] - 0.69) < 0.05  # the F-0005 calibration number
    assert body["epr"] is not None
    assert abs(sum(body["sigma_a"]) - 1.0) < 1e-8
    assert len(body["allocations_a"]) == len(body["sigma_a"])


def test_blotto_large_budget_omits_epr_with_warning():
    r = client.post("/v1/domains/blotto/read", json={"budget_a": 7, "budget_b": 7})
    assert r.status_code == 200
    body = r.json()
    assert body["epr"] is None
    assert body["warnings"]
    assert body["alpha"] > 0.0


# ---------------------------------------------------------------------------
# /v1/diagnose — the whole-system verdict, with every caveat that travels with it
# ---------------------------------------------------------------------------

_COORD_KEYS = {"name", "value", "lo", "hi", "kind", "method", "warnings"}


def test_diagnose_game_route_returns_the_full_diagnosis_shape():
    body = client.post("/v1/diagnose", json={"payoffs": RPS["payoffs"], "lam": 1.5}).json()
    assert body["quadrant"] == "whirlpool"
    assert body["live_quadrants"] == ["whirlpool"]
    for coord in ("response", "dissipation"):
        assert set(body[coord]) >= _COORD_KEYS
        assert body[coord]["kind"] in ("point", "interval", "upper_bound", "lower_bound", "absent")
        assert body[coord]["method"]
    assert body["alpha"] > 0.9
    assert body["lam"] == 1.5
    assert body["tier"] == "certified"
    assert body["warnings"]  # the lambda-scaling warning must travel over HTTP
    assert body["refusals"] == []
    assert body["provenance"]["library_version"]
    assert body["provenance"]["payoff_range"] == 2.0
    assert body["provenance"]["lambda_normalised"] == 3.0
    assert "snippet" in body and "strataq" in body["snippet"]
    assert "WHIRLPOOL" in body["headline"]


def test_diagnose_potential_game_reads_landscape_with_zero_dissipation():
    body = client.post("/v1/diagnose", json={"payoffs": COORD["payoffs"], "lam": 0.4}).json()
    assert body["quadrant"] == "landscape"
    assert body["response"]["kind"] == "point"
    assert body["dissipation"]["value"] < 1e-12
    assert body["alpha"] < 1e-9


def test_diagnose_refuses_R_at_criticality_as_a_bound():
    """The same potential game at lambda = 1.0 sits on its pitchfork: R is undefined
    there, so the endpoint must refuse it as an absent coordinate with a live quadrant
    set -- never return the rounding noise as a reading."""
    body = client.post("/v1/diagnose", json={"payoffs": COORD["payoffs"], "lam": 1.0}).json()
    assert body["quadrant"] == "undetermined"
    assert body["response"]["kind"] == "absent"
    assert body["refusals"] and any("criticality" in r for r in body["refusals"])
    assert len(body["live_quadrants"]) == 2
    assert body["provenance"]["rho_SB"] >= 1.0 - 1e-3


def test_diagnose_readings_route_refuses_as_bounds_not_errors():
    body = client.post("/v1/diagnose", json={"chi": [[1.0, 0.05], [0.01, 1.0]]}).json()
    assert body["quadrant"] == "undetermined"
    assert len(body["live_quadrants"]) > 1
    assert body["dissipation"]["kind"] == "absent"
    assert body["refusals"]  # no chi_se, no series: both are stated, neither raises
    assert any("chi_se" in r for r in body["refusals"])


def test_diagnose_series_route_gives_a_one_sided_epr_bound():
    import numpy as np

    series = list(np.cumsum(np.random.default_rng(3).normal(size=400)))
    body = client.post(
        "/v1/diagnose", json={"series": series, "n_surrogates": 60, "seed": 1}
    ).json()
    assert body["response"]["kind"] == "absent"
    assert body["dissipation"]["kind"] in ("upper_bound", "lower_bound")
    assert body["provenance"]["n_series"] == 400


def test_diagnose_guards():
    assert client.post("/v1/diagnose", json={}).status_code == 422
    assert client.post("/v1/diagnose", json={"payoffs": RPS["payoffs"]}).status_code == 422
    assert client.post("/v1/diagnose", json={"chi": [[1.0, 0.1]]}).status_code == 422
    big = {"payoffs": [[[0.0] * 20 for _ in range(20)]] * 2, "lam": 1.0}
    assert client.post("/v1/diagnose", json=big).status_code == 413


def test_diagnose_dense_generator_guard():
    """3 x 8 actions is inside the per-player cap but past the dense EPR guard (512 > 400)."""
    payoffs = [[[[0.1 * (i + j + k) for k in range(8)] for j in range(8)] for i in range(8)]] * 3
    assert client.post("/v1/diagnose", json={"payoffs": payoffs, "lam": 1.0}).status_code == 413


# ---------------------------------------------------------------------------
# /v1/fit — the estimation workflow, tidy data in
# ---------------------------------------------------------------------------

FIT_GAME = [
    [[3.0, 0.0, 1.5], [1.0, 2.0, 0.5], [0.0, 1.0, 2.5]],
    [[2.0, 1.0, 0.0], [0.5, 3.0, 1.0], [1.5, 0.0, 2.0]],
]


def _fit_counts(lam, n, seed):
    import jax
    from strataq.estimate.lam import sample_choices
    from strataq.finite.games.tensor import DenseTensorGame

    game = DenseTensorGame([jnp.asarray(u) for u in FIT_GAME])
    return [[int(x) for x in c] for c in sample_choices(game, lam, n, jax.random.key(seed))]


def _tidy_rows(lam, n, seed, treatment):
    import numpy as np
    from strataq.core.solve.fixedpoint import logit_qre
    from strataq.finite.games.tensor import DenseTensorGame

    game = DenseTensorGame([jnp.asarray(u) for u in FIT_GAME])
    sigma = [np.asarray(s, dtype=float) for s in logit_qre(game, lam).sigma]
    rng = np.random.default_rng(seed)
    rows = []
    for p, s in enumerate(sigma):
        for i, a in enumerate(rng.choice(len(s), size=n, p=s / s.sum())):
            rows.append(
                {
                    "subject": f"p{p}s{i % 25}",
                    "player": p,
                    "action": int(a),
                    "treatment": treatment,
                }
            )
    return rows


def test_fit_from_counts_recovers_lambda_with_a_named_interval():
    body = client.post(
        "/v1/fit",
        json={
            "payoffs": FIT_GAME,
            "counts": _fit_counts(1.2, 20_000, 11),
            "ci": "profile",
            "n_grid": 60,
        },
    ).json()
    assert abs(body["lam_hat"] - 1.2) / 1.2 < 0.15
    assert body["ci_low"] <= 1.2 <= body["ci_high"]
    assert "profile likelihood" in body["ci_method"]
    assert body["identified"] is True
    assert body["n_obs"] == 40_000
    assert body["loglik"] < 0
    assert body["lr_uniform"]["df"] == 1 and body["lr_uniform"]["p"] < 1e-10
    assert body["lr_nash"]["p"] < 1e-10
    assert "boundary" in body["lr_uniform"]["note"]
    assert body["warnings"]
    assert "lambda_hat" in body["summary"]
    assert body["provenance"]["library_version"]
    assert body["provenance"]["payoff_range"] == 3.0


def test_fit_from_tidy_data_keeps_subjects_and_splits_by_treatment():
    rows = _tidy_rows(0.5, 3_000, 41, "low") + _tidy_rows(3.0, 3_000, 42, "high")
    body = client.post(
        "/v1/fit",
        json={
            "payoffs": FIT_GAME,
            "data": rows,
            "by": "treatment",
            "n_grid": 60,
            "n_boot": 80,
            "seed": 2,
        },
    ).json()
    assert body["n_subjects"] == 50
    assert "cluster bootstrap on subject" in body["ci_method"]
    keyed = {g["key"]: g for g in body["groups"]}
    assert set(keyed) == {"low", "high"}
    assert keyed["low"]["lam_hat"] < keyed["high"]["lam_hat"]
    assert body["homogeneity"]["p"] < 1e-3
    assert body["provenance"]["clustered_on"] == "subject"


def test_fit_refuses_to_quote_a_flat_likelihood():
    body = client.post(
        "/v1/fit",
        json={
            "payoffs": RPS["payoffs"],
            "counts": [[13400, 13300, 13300], [13350, 13350, 13300]],
            "n_grid": 60,
            "n_boot": 40,
        },
    ).json()
    assert body["lam_hat"] is None
    assert body["identified"] is False
    assert body["kind"] == "unidentified"
    assert body["refusals"] and "NOT IDENTIFIED" in body["refusals"][0]
    assert body["ci_low"] == 0.05 and body["ci_high"] == 20.0


def test_fit_guards():
    counts = [[10, 10, 10], [10, 10, 10]]
    assert client.post("/v1/fit", json={"payoffs": FIT_GAME}).status_code == 422
    assert (
        client.post(
            "/v1/fit",
            json={"payoffs": FIT_GAME, "counts": counts, "data": [{"player": 0, "action": 0}]},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/fit", json={"payoffs": FIT_GAME, "counts": [[0, 0, 0], [0, 0, 0]]}
        ).status_code
        == 422
    )
    assert (
        client.post("/v1/fit", json={"payoffs": FIT_GAME, "counts": [[1, 1, 1]]}).status_code == 422
    )
    big = {"payoffs": [[[0.0] * 20 for _ in range(20)]] * 2, "counts": [[1] * 20, [1] * 20]}
    assert client.post("/v1/fit", json=big).status_code == 413
    assert (
        client.post(
            "/v1/fit", json={"payoffs": FIT_GAME, "counts": counts, "by": "treatment"}
        ).status_code
        == 422
    )
