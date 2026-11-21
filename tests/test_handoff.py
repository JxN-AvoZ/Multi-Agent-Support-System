"""Human handoff: ticket lifecycle, operator endpoints, resume via history."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import main as app

KEY = "k" * 32
OPERATOR = {"X-Operator-Key": KEY}


@pytest.fixture(autouse=True)
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "TICKET_DB_FILE", tmp_path / "test.sqlite3")
    monkeypatch.setattr(app, "AUTH_MODE", "demo")
    monkeypatch.setattr(app, "OPERATOR_API_KEY", KEY)
    monkeypatch.setattr(app, "REPLAN_ENABLED", False)
    monkeypatch.setattr(app, "users", lambda: {"cliente1988": {}, "other": {}})


@pytest.fixture
def client():
    return TestClient(app.app)


def unresolved(monkeypatch):
    """Agents that cannot resolve anything, so every turn escalates."""
    seen = {"histories": []}

    def fake_route(message, history=""):
        seen["histories"].append(history)
        return {"plan": ["knowledge"], "angry": False, "human": False,
                "safe": True, "routing_error": False}

    monkeypatch.setattr(app, "route", fake_route)
    monkeypatch.setattr(
        app, "knowledge_agent",
        lambda message, notes="": app.result("knowledge", "I couldn't verify that.", 0.1),
    )
    return seen


def escalate(client, monkeypatch):
    seen = unresolved(monkeypatch)
    first = client.post("/chat", json={"message": "my problem", "user_id": "cliente1988"}).json()
    ticket_id = first["answer"].split("ticket ")[1].rstrip(").")
    return seen, first["session_id"], ticket_id


def act(client, ticket_id, verb, **body):
    body.setdefault("operator_id", "ana")
    return client.post(f"/operator/tickets/{ticket_id}/{verb}", json=body, headers=OPERATOR)


def test_full_loop_human_reply_reaches_the_next_turn(client, monkeypatch):
    seen, session, ticket = escalate(client, monkeypatch)

    assert act(client, ticket, "claim").json()["status"] == "claimed"
    replied = act(client, ticket, "reply", message="We reset your terminal remotely.")
    assert replied.json()["status"] == "replied"

    mine = client.get(f"/tickets/{ticket}", params={"user_id": "cliente1988"}).json()
    assert mine["status"] == "replied"
    assert mine["operator_reply"] == "We reset your terminal remotely."

    client.post("/chat", json={"message": "thanks, it works", "user_id": "cliente1988",
                               "session_id": session})
    last_history = seen["histories"][-1]
    assert "<operator_reply>We reset your terminal remotely.</operator_reply>" in last_history

    assert act(client, ticket, "close").json()["status"] == "closed"


def test_operator_routes_need_the_key(client, monkeypatch):
    _, _, ticket = escalate(client, monkeypatch)

    assert client.get("/operator/tickets").status_code == 401
    assert client.get("/operator/tickets", headers={"X-Operator-Key": "x" * 32}).status_code == 401
    assert client.post(f"/operator/tickets/{ticket}/claim", json={"operator_id": "ana"}).status_code == 401


def test_operator_routes_are_off_when_no_key_is_configured(client, monkeypatch):
    monkeypatch.setattr(app, "OPERATOR_API_KEY", None)

    assert client.get("/operator/tickets", headers=OPERATOR).status_code == 503


def test_two_operators_cannot_claim_the_same_ticket(client, monkeypatch):
    _, _, ticket = escalate(client, monkeypatch)

    assert act(client, ticket, "claim", operator_id="ana").status_code == 200
    assert act(client, ticket, "claim", operator_id="ana").status_code == 200  # idempotent
    assert act(client, ticket, "claim", operator_id="bia").status_code == 409


def test_reply_requires_claiming_first_and_by_the_same_operator(client, monkeypatch):
    _, _, ticket = escalate(client, monkeypatch)

    assert act(client, ticket, "reply", message="hi").status_code == 409  # not claimed
    act(client, ticket, "claim", operator_id="ana")
    assert act(client, ticket, "reply", operator_id="bia", message="hi").status_code == 409
    assert act(client, ticket, "reply", operator_id="ana", message="hi").status_code == 200


def test_a_closed_ticket_takes_no_more_replies(client, monkeypatch):
    _, _, ticket = escalate(client, monkeypatch)
    act(client, ticket, "claim")
    act(client, ticket, "close")

    assert act(client, ticket, "reply", message="late").status_code == 409


def test_retrying_a_reply_with_the_same_key_is_idempotent(client, monkeypatch):
    _, session, ticket = escalate(client, monkeypatch)
    act(client, ticket, "claim")

    first = act(client, ticket, "reply", message="Done.", idempotency_key="r-1").json()
    retry = act(client, ticket, "reply", message="Done.", idempotency_key="r-1").json()

    assert first["duplicate"] is False and retry["duplicate"] is True
    with app._connect_ticket_db() as connection:
        replies = connection.execute("SELECT COUNT(*) FROM ticket_replies").fetchone()[0]
        turns = connection.execute(
            "SELECT COUNT(*) FROM messages WHERE role = 'operator' AND session_id = ?", (session,)
        ).fetchone()[0]
    assert replies == 1 and turns == 1


def test_operator_text_is_redacted_before_the_customer_sees_it(client, monkeypatch):
    _, _, ticket = escalate(client, monkeypatch)
    act(client, ticket, "claim")

    act(client, ticket, "reply", message="Was it card 4111 1111 1111 1111?")

    mine = client.get(f"/tickets/{ticket}", params={"user_id": "cliente1988"}).json()
    assert "4111" not in mine["operator_reply"] and "[CARD]" in mine["operator_reply"]


def test_a_customer_cannot_read_someone_elses_ticket(client, monkeypatch):
    _, _, ticket = escalate(client, monkeypatch)

    response = client.get(f"/tickets/{ticket}", params={"user_id": "other"})

    assert response.status_code == 404


def test_operator_sees_the_queue_and_the_conversation(client, monkeypatch):
    _, _, ticket = escalate(client, monkeypatch)

    queue = client.get("/operator/tickets", params={"status": "open"}, headers=OPERATOR).json()
    detail = client.get(f"/operator/tickets/{ticket}", headers=OPERATOR).json()

    assert [t["ticket_id"] for t in queue["tickets"]] == [ticket]
    assert detail["conversation"][0] == {"role": "customer", "content": "my problem"}
    assert "user_id" not in detail
    assert client.get("/operator/tickets", params={"status": "bogus"}, headers=OPERATOR).status_code == 422


def test_unknown_ticket_is_404(client):
    assert act(client, "TCK-NOPE", "claim").status_code == 404
