from fastapi.testclient import TestClient

from backend.main import app


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["message"]


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code in (200, 503)


def test_transaction_history_endpoint():
    with TestClient(app) as client:
        response = client.get("/transactions/history?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "transactions" in data