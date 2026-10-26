from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)
GOOD = {
    "amount": 1899, "avg_amount_30d": 35, "channel": "online", "merchant_category": "electronics",
    "country": "NG", "card_present": False, "account_age_days": 12, "num_txn_24h": 9,
    "distance_from_home_km": 6400, "timestamp": "2024-06-15T02:40:00",
}


def test_health():
    assert client.get("/health").status_code == 200


def test_negative_amount_is_rejected_with_a_clear_error():
    r = client.post("/predict", json={**GOOD, "amount": -5})
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_input"


@pytest.mark.skipif(not Path("model.joblib").exists(), reason="train first: python -m app.train")
def test_predict_returns_a_probability():
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    assert 0 <= r.json()["fraud_probability"] <= 1


@pytest.mark.skipif(not Path("model.joblib").exists(), reason="train first: python -m app.train")
def test_optional_fields_can_be_left_out():
    required = {k: GOOD[k] for k in ["amount", "channel", "merchant_category", "country", "card_present",
                                     "account_age_days", "num_txn_24h"]}
    assert client.post("/predict", json=required).status_code == 200


def test_too_many_requests_are_limited():
    codes = [client.post("/predict", json=GOOD).status_code for _ in range(35)]
    assert 429 in codes
    assert client.post("/predict", json=GOOD).json()["error"] == "rate_limited"
