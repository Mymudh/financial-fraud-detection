
from fastapi.testclient import TestClient

from backend.main import app


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200

    data = response.json()
    assert data["message"] == "Financial Fraud Detection API is running"
    assert data["version"] == "1.6.0"
    assert data["docs"] == "/docs"
    assert data["health"] == "/health"
    assert data["transaction_history"] == "/transactions/history"


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code in (200, 503)
    assert response.headers["content-type"].startswith("application/json")

    data = response.json()
    assert isinstance(data, dict)


def test_transaction_history_endpoint():
    with TestClient(app) as client:
        response = client.get("/transactions/history?limit=10")

    assert response.status_code == 200

    data = response.json()
    assert "count" in data
    assert "transactions" in data
    assert isinstance(data["count"], int)
    assert isinstance(data["transactions"], list)
    assert data["count"] == len(data["transactions"])

