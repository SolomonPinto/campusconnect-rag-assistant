from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_missing_message_returns_structured_error() -> None:
    response = client.post("/api/chat", json={"sessionId": "demo-session"})

    assert response.status_code == 422
    assert response.json() == {"error": "Message field is required."}


def test_invalid_json_returns_structured_error() -> None:
    response = client.post(
        "/api/chat",
        content="{not-valid-json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json() == {"error": "Invalid JSON payload."}
