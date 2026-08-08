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
