"""Bounded replanning and the independent groundedness verifier."""

from __future__ import annotations

import pytest

import main as app


@pytest.fixture(autouse=True)
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "TICKET_DB_FILE", tmp_path / "test.sqlite3")
    monkeypatch.setattr(app, "REPLAN_ENABLED", True)
    monkeypatch.setattr(app, "MAX_REPLANS", 1)


def route_to(monkeypatch, *plan):
    monkeypatch.setattr(
        app, "route",
        lambda message, history="": {"plan": list(plan), "angry": False, "human": False,
                                      "safe": True, "routing_error": False},
    )


def fake_agents(monkeypatch, *, support=None, knowledge=None, ran=None):
    ran = ran if ran is not None else []
    support = support or app.result(
        "support", "Your terminal is offline.", 0.85,
        calls=[{"tool": "get_terminal_status", "output": {"connection": "offline"}}],
    )
    knowledge = knowledge or app.result("knowledge", "Check the Wi-Fi.", 0.9)

    def run_support(message, uid, history=""):
        ran.append("support")
        return support

    def run_knowledge(message, notes=""):
        ran.append("knowledge")
        ran.append(("notes", notes))
        return knowledge

    monkeypatch.setattr(app, "support_agent", run_support)
    monkeypatch.setattr(app, "knowledge_agent", run_knowledge)
    monkeypatch.setattr(app, "ask", lambda *a, **k: "Merged answer")
    return ran


def reviewer(monkeypatch, decision):
    calls = []

    def fake_ask_json(system, user, model=None):
        calls.append((system, user))
        return decision

    monkeypatch.setattr(app, "ask_json", fake_ask_json)
    return calls


def test_support_result_can_add_knowledge_and_passes_findings_on(monkeypatch):
    route_to(monkeypatch, "support")
    ran = fake_agents(monkeypatch)
    calls = reviewer(monkeypatch, {"action": "add_knowledge", "reason": "terminal offline"})

    response, trace = app.handle_traced("my machine is dead", "cliente1988")

    assert response["agents_used"] == ["router", "support", "knowledge"]
    assert trace["replans"] == [
        {"after": "support", "action": "add_knowledge", "applied": True, "reason": "terminal offline"}
    ]
    assert "Your terminal is offline." in ran[-1][1]  # notes carried the support finding
    assert calls[0][0] == app.prompts.REPLAN
    assert 'tools="get_terminal_status:ok"' in calls[0][1]


def test_reviewer_can_decide_to_finish(monkeypatch):
    route_to(monkeypatch, "support")
    ran = fake_agents(monkeypatch)
    reviewer(monkeypatch, {"action": "finish", "reason": "answered"})

    response, trace = app.handle_traced("show my last sale", "cliente1988")

    assert response["agents_used"] == ["router", "support"]
    assert "knowledge" not in ran
    assert trace["replans"][0]["applied"] is False


def test_low_confidence_knowledge_can_add_support(monkeypatch):
    route_to(monkeypatch, "knowledge")
    ran = fake_agents(
        monkeypatch,
        knowledge=app.result("knowledge", "I couldn't find a reliable answer.", 0.0),
    )
    reviewer(monkeypatch, {"action": "add_support", "reason": "needs account data"})

    response, trace = app.handle_traced("why was my sale rejected", "cliente1988")

    assert ran[0] == "knowledge" and "support" in ran
    assert response["agents_used"][:3] == ["router", "knowledge", "support"]
    assert trace["replans"][0]["applied"] is True


@pytest.mark.parametrize("bad", [{}, {"action": "launch_missiles"}, {"action": 5}, {"nope": 1}])
def test_unusable_reviewer_output_means_finish(monkeypatch, bad):
    route_to(monkeypatch, "support")
    fake_agents(monkeypatch)
    reviewer(monkeypatch, bad)

    response, trace = app.handle_traced("show my last sale", "cliente1988")

    assert response["agents_used"] == ["router", "support"]
    assert trace["replans"][0]["applied"] is False


def test_both_agents_planned_means_no_review(monkeypatch):
    route_to(monkeypatch, "support", "knowledge")
    fake_agents(monkeypatch)
    calls = reviewer(monkeypatch, {"action": "add_knowledge"})

    app.handle("my machine is dead", "cliente1988")

    assert calls == []


def test_support_tool_errors_are_not_reviewed(monkeypatch):
    route_to(monkeypatch, "support")
    fake_agents(
        monkeypatch,
        support=app.result("support", "I could not read it.", 0.2,
                           calls=[{"tool": "get_terminal_status", "output": {"error": "down"}}]),
    )
    calls = reviewer(monkeypatch, {"action": "add_knowledge"})

    app.handle("my terminal?", "cliente1988")

    assert calls == []


def test_an_agent_never_runs_twice_and_replans_are_bounded(monkeypatch):
    route_to(monkeypatch, "support")
    ran = fake_agents(monkeypatch)
    calls = reviewer(monkeypatch, {"action": "add_knowledge"})

    app.handle("my machine is dead", "cliente1988")

    assert len(calls) == 1
    assert [r for r in ran if isinstance(r, str)] == ["support", "knowledge"]


def test_replanning_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(app, "REPLAN_ENABLED", False)
    route_to(monkeypatch, "support")
    fake_agents(monkeypatch)
    calls = reviewer(monkeypatch, {"action": "add_knowledge"})

    app.handle("my machine is dead", "cliente1988")

    assert calls == []


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

PASSAGE = {"url": "https://site.getnet.com.br/pix/", "text": "O Pix permite receber vendas na hora."}


def knowledge_with(monkeypatch, verify, *, answer_confidence=0.9):
    monkeypatch.setattr(app, "CHUNKS", [PASSAGE])
    app.build_retrieval_stats(app.CHUNKS)
    monkeypatch.setattr(app, "web_search", lambda q: [])
    seen = {"verify": None}

    def fake_ask_json(system, user, model=None):
        if system.startswith(app.UNTRUSTED_CONTEXT_NOTICE) and app.prompts.VERIFY in system:
            seen["verify"] = (user, model)
            return verify
        if system.startswith(app.prompts.REWRITE):
            return {"queries": ["pix receber vendas"], "use_web": False}
        return {"answer": "O Pix recebe vendas na hora [1].", "grounded": True,
                "confidence": answer_confidence}

    monkeypatch.setattr(app, "ask_json", fake_ask_json)
    return seen


def test_a_supported_answer_is_accepted_and_marked_verified(monkeypatch):
    seen = knowledge_with(monkeypatch, {"supported": True, "unsupported_claims": [], "score": 0.95})

    outcome = app.knowledge_agent("Como receber com Pix?")

    assert outcome["verified"] is True
    assert outcome["confidence"] == 0.9  # min(answer confidence, verifier score)
    assert outcome["sources"] == [PASSAGE["url"]]
    user, model = seen["verify"]
    assert "<passages>" in user and PASSAGE["text"] in user and "<answer>" in user
    assert model == app.VERIFY_MODEL


def test_verifier_score_lowers_confidence(monkeypatch):
    knowledge_with(monkeypatch, {"supported": True, "unsupported_claims": [], "score": 0.82})

    assert app.knowledge_agent("Como receber com Pix?")["confidence"] == 0.82


def test_an_unsupported_answer_is_rejected(monkeypatch):
    knowledge_with(
        monkeypatch,
        {"supported": False, "unsupported_claims": ["fee is 1%"], "score": 0.4},
    )

    outcome = app.knowledge_agent("Como receber com Pix?")

    assert outcome["verified"] is False
    assert outcome["confidence"] == 0.0
    assert "couldn't find a reliable answer" in outcome["answer"]
    assert outcome["sources"] == []


def test_a_low_score_is_rejected_even_if_marked_supported(monkeypatch):
    knowledge_with(monkeypatch, {"supported": True, "unsupported_claims": [], "score": 0.5})

    assert app.knowledge_agent("Como receber com Pix?")["verified"] is False


def test_verifier_outage_keeps_the_answer_but_caps_confidence(monkeypatch):
    knowledge_with(monkeypatch, {})

    outcome = app.knowledge_agent("Como receber com Pix?")

    assert outcome["verified"] is None
    assert outcome["confidence"] == app.VERIFIER_UNAVAILABLE_CONFIDENCE_CAP
    assert "Pix" in outcome["answer"]


def test_malformed_verifier_output_is_an_outage_not_a_pass(monkeypatch):
    knowledge_with(monkeypatch, {"supported": "yes please", "score": 7})

    outcome = app.knowledge_agent("Como receber com Pix?")

    assert outcome["verified"] is None
    assert outcome["confidence"] <= app.VERIFIER_UNAVAILABLE_CONFIDENCE_CAP


def test_verification_can_be_switched_off(monkeypatch):
    seen = knowledge_with(monkeypatch, {"supported": False, "score": 0.0})
    monkeypatch.setattr(app, "VERIFY_GROUNDING", False)

    outcome = app.knowledge_agent("Como receber com Pix?")

    assert seen["verify"] is None and outcome["verified"] is None
    assert outcome["confidence"] == 0.9


def test_a_rejected_answer_escalates_the_request(monkeypatch):
    knowledge_with(monkeypatch, {"supported": False, "unsupported_claims": ["x"], "score": 0.1})
    route_to(monkeypatch, "knowledge")
    monkeypatch.setattr(app, "REPLAN_ENABLED", False)

    response, trace = app.handle_traced("Como receber com Pix?", "cliente1988")

    assert response["escalated"] is True and "TCK-" in response["answer"]
    assert trace["verified"] is False
