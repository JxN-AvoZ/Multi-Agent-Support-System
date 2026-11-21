"""Whole-stack test over HTTP with a scripted stand-in for the Google GenAI client.

It cannot show that the real model behaves; it shows that every piece of our
own code (auth, router, tool loop, knowledge, verifier, compose, memory,
escalation, operator loop, metrics) works together when the model returns
well-formed output.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import jwt
import pytest
from fastapi.testclient import TestClient

import main as app
import prompts

SECRET = "k" * 48
OPERATOR_KEY = "o" * 32


def text_response(text):
    return SimpleNamespace(
        text=text,
        function_calls=[],
        candidates=[
            SimpleNamespace(
                finish_reason="STOP",
                content=SimpleNamespace(parts=[SimpleNamespace(text=text)]),
            )
        ],
        usage_metadata=SimpleNamespace(prompt_token_count=50, candidates_token_count=10),
    )


class ScriptedModels:
    """Answers by recognising which system prompt it was called with."""

    def __init__(self):
        self.calls = []
        self.router_plan = ["support", "knowledge"]
        self.support_step = 0

    def generate_content(self, *, model, contents, config):
        system = getattr(config, "system_instruction", "") or ""
        if isinstance(contents, str):
            user = contents
            has_tool_response = False
        else:
            user = ""
            has_tool_response = any(
                getattr(part, "function_response", None) is not None
                for turn in contents
                for part in (getattr(turn, "parts", None) or [])
            )
        self.calls.append((system[:40], user))

        if system.startswith(prompts.ROUTER[:40]):
            return text_response(json.dumps({
                "plan": self.router_plan, "sentiment": "neutral",
                "human_requested": "person" in user.lower(), "safe": True}))
        if system.startswith(prompts.SUPPORT[:40]):
            if has_tool_response:
                return text_response("Your terminal is offline (NO_CONNECTIVITY).")
            fc = SimpleNamespace(name="get_terminal_status", args={})
            return SimpleNamespace(
                text="",
                function_calls=[fc],
                candidates=[
                    SimpleNamespace(
                        finish_reason="STOP",
                        content=SimpleNamespace(role="model", parts=[SimpleNamespace(function_call=fc)]),
                    )
                ],
                usage_metadata=SimpleNamespace(prompt_token_count=60, candidates_token_count=12),
            )
        if prompts.REWRITE[:40] in system:
            return text_response(json.dumps({"use_web": False, "queries": ["wi-fi maquininha rede"]}))
        if prompts.ANSWER[:40] in system:
            return text_response(json.dumps({"grounded": True, "confidence": 0.9,
                                             "answer": "Connect it to Wi-Fi in the settings menu [1]."}))
        if prompts.VERIFY[:40] in system:
            return text_response(json.dumps({"supported": True, "unsupported_claims": [], "score": 0.95}))
        if prompts.COMPOSE[:40] in system:
            history_seen = "<conversation_history>" in user
            return text_response("Your terminal is offline. Connect it to Wi-Fi in settings."
                                 + (" (earlier context used)" if history_seen else ""))
        if prompts.SUMMARY[:40] in system:
            return text_response("Customer had a terminal connectivity problem.")
        if prompts.REPLAN[:40] in system:
            return text_response(json.dumps({"action": "finish", "reason": "enough"}))
        raise AssertionError(f"unexpected prompt: {system[:80]}")


@pytest.fixture
def stack(monkeypatch, tmp_path):
    models = ScriptedModels()
    monkeypatch.setattr(app, "get_genai_client", lambda: SimpleNamespace(models=models))
    monkeypatch.setattr(app, "TICKET_DB_FILE", tmp_path / "t.sqlite3")
    monkeypatch.setattr(app, "AUTH_MODE", "jwt")
    monkeypatch.setattr(app, "JWT_SECRET", SECRET)
    monkeypatch.setattr(app, "JWT_ISSUER", None)
    monkeypatch.setattr(app, "JWT_AUDIENCE", None)
    monkeypatch.setattr(app, "OPERATOR_API_KEY", OPERATOR_KEY)
    monkeypatch.setattr(app, "CHUNKS", [{"url": "https://site.getnet.com.br/wifi",
                                         "text": "Para conectar a maquininha ao Wi-Fi abra as configuracoes e escolha a rede. " * 3}])
    app.build_retrieval_stats(app.CHUNKS)
    app.METRICS.reset()
    token = jwt.encode({"sub": "cliente1988", "exp": int(time.time()) + 600}, SECRET, algorithm="HS256")
    client = TestClient(app.app)
    client.headers.update({"Authorization": f"Bearer {token}"})
    return SimpleNamespace(model=models, client=client)


def test_two_agents_answer_and_the_second_turn_remembers(stack):
    first = stack.client.post("/chat", json={"message": "My card machine won't connect to the internet"})
    body = first.json()

    assert first.status_code == 200
    assert body["agents_used"] == ["router", "support", "knowledge"]
    assert body["sources"] == ["https://site.getnet.com.br/wifi"]
    assert "offline" in body["answer"] and body["escalated"] is False
    session = body["session_id"]

    second = stack.client.post("/chat", json={"message": "and how do I fix it?", "session_id": session})

    assert second.json()["session_id"] == session
    assert "earlier context used" in second.json()["answer"]


def test_escalation_then_operator_reply_reaches_the_customer(stack):
    stack.model.router_plan = ["knowledge"]
    first = stack.client.post("/chat", json={"message": "I want to talk to a person"}).json()
    ticket_id = first["answer"].split("ticket ")[1].split(")")[0]
    assert first["escalated"] is True and first["agents_used"][-1] == "escalation"
    session = first["session_id"]

    operator = {"X-Operator-Key": OPERATOR_KEY}
    # The customer's own bearer token must not open operator routes.
    assert stack.client.get("/operator/tickets").status_code == 401
    anonymous = TestClient(app.app)
    assert anonymous.get("/operator/tickets", headers=operator).status_code == 200
    assert anonymous.post(f"/operator/tickets/{ticket_id}/claim", headers=operator,
                          json={"operator_id": "ana"}).json()["status"] == "claimed"
    reply = {"operator_id": "ana", "message": "Limit raised, retry after 9am", "idempotency_key": "r1"}
    assert anonymous.post(f"/operator/tickets/{ticket_id}/reply", headers=operator, json=reply).status_code == 200
    assert anonymous.post(f"/operator/tickets/{ticket_id}/reply", headers=operator, json=reply).status_code == 200

    mine = stack.client.get(f"/tickets/{ticket_id}").json()
    assert mine["status"] == "replied" and "9am" in mine["operator_reply"]

    stack.model.router_plan = ["knowledge"]
    stack.client.post("/chat", json={"message": "any news?", "session_id": session})
    assert any("Limit raised" in user for _system, user in stack.model.calls), \
        "the operator's reply never reached a model prompt"


def test_another_customers_token_cannot_use_my_session(stack):
    session = stack.client.post("/chat", json={"message": "hello there"}).json()["session_id"]
    other = jwt.encode({"sub": "someone-else", "exp": int(time.time()) + 600}, SECRET, algorithm="HS256")

    response = TestClient(app.app).post(
        "/chat", json={"message": "hi", "session_id": session},
        headers={"Authorization": f"Bearer {other}"})

    assert response.status_code == 403


def test_metrics_reflect_traffic_and_tokens(stack):
    stack.client.post("/chat", json={"message": "My card machine won't connect"})

    snapshot = stack.client.get("/metrics").json()

    assert snapshot["requests"] == 1
    assert snapshot["input_tokens"] > 0 and snapshot["latency_ms_p95"] is not None
