from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_login_and_chat_pages_are_served() -> None:
    login_response = client.get("/")
    chat_response = client.get("/chat")

    assert login_response.status_code == 200
    assert "Welcome back" in login_response.text
    assert chat_response.status_code == 200
    assert "Knowledge Base Assistant" in chat_response.text


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
