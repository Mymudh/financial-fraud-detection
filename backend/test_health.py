
import pytest
import httpx

from backend.main import app


@pytest.mark.anyio
async def test_root_endpoint():
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test"
    ) as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert response.json()["message"]


@pytest.mark.anyio
async def test_health_endpoint():
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code in (200, 503)


@pytest.mark.anyio
async def test_transaction_history_endpoint():
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test"
    ) as client:
        response = await client.get("/transactions/history?limit=10")

    assert response.status_code == 200

    data = response.json()

    assert "count" in data
    assert "transactions" in data

