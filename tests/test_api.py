from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "AI Helpdesk Ticket Assistant API is running.",
        "status": "ok",
    }


def test_empty_ticket_is_rejected():

    response = client.post(
        "/analyze-ticket",
        json={
            "text": "   ",
        },
    )

    assert response.status_code == 422


def test_missing_text_is_rejected():

    response = client.post(
        "/analyze-ticket",
        json={},
    )

    assert response.status_code == 422

def test_analyze_ticket_endpoint(monkeypatch):

    fake_result = {
        "category": "login problem",
        "category_confidence": 0.91,
        "priority": "medium priority",
        "priority_confidence": 0.82,
        "summary": "I cannot log in.",
        "suggested_response": (
            "Please confirm whether an error message appears during login."
        ),
        "requires_human_review": False,
    }

    def fake_get_classifier():
        return "fake-classifier"

    def fake_analyze_ticket(
        classifier,
        text: str,
    ):
        return fake_result

    monkeypatch.setattr(
        "app.main.get_classifier",
        fake_get_classifier,
    )

    monkeypatch.setattr(
        "app.main.analyze_ticket_with_ai",
        fake_analyze_ticket,
    )

    response = client.post(
        "/analyze-ticket",
        json={
            "text": "I cannot log in.",
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["category"] == "login problem"
    assert result["priority"] == "medium priority"
    assert result["category_confidence"] == 0.91