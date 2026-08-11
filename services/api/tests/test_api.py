"""API surface tests: every endpoint against calibration-known games."""

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
