"""The evaluation runner's scoring logic, with the model and handler replaced."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import agentic_eval as ev
import eval_metrics as m
import main as app


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "TICKET_DB_FILE", tmp_path / "t.sqlite3")


CASE = {
    "id": "terminal",
    "expected_agents": ["support", "knowledge"],
    "expected_tools": ["get_terminal_status", "Tavily search"],
    "required_facts": ["offline", "wi-?fi"],
}


def run(plan, executed, tools, answer, **extra):
    response = {"answer": answer, "escalated": False, "confidence": 0.9, "latency_ms": 10}
    trace = {"plan": plan, "executed": executed, "tool_calls": tools, "usage": {}, **extra}
    return ev.score_case(CASE, response, trace)


def test_a_perfect_run_scores_everything_right():
    row = run(["support", "knowledge"], ["support", "knowledge"],
              ["get_terminal_status"], "Your terminal is offline. Check the Wi-Fi.")

    assert row["routing_ok"] and row["final_agents_ok"]
    assert row["tool_recall"] == 1.0 and row["tool_precision"] == 1.0
    assert row["fact_coverage"] == 1.0 and row["missing_facts"] == []


def test_non_tool_expectations_are_ignored_for_tool_scoring():
    assert ev.real_tools(["get_terminal_status", "Tavily search"]) == ["get_terminal_status"]


def test_a_replan_can_fix_the_final_agents_but_not_the_routing_score():
    row = run(["support"], ["support", "knowledge"], ["get_terminal_status"], "offline wifi")

    assert row["routing_ok"] is False   # the router's own plan was wrong
    assert row["final_agents_ok"] is True  # but the workflow recovered


def test_wrong_tools_and_missing_facts_are_reported():
    row = run(["support", "knowledge"], ["support", "knowledge"],
              ["get_recent_transactions"], "All good.")

    assert row["tool_recall"] == 0.0 and row["tool_precision"] == 0.0
    assert row["missing_facts"] == ["offline", "wi-?fi"]


def test_summary_averages_and_judge(monkeypatch):
    rows = [
        {**run(["support", "knowledge"], ["support", "knowledge"], ["get_terminal_status"], "offline wifi"),
         "judge": {"correctness": 5, "groundedness": 4, "helpfulness": 3}},
        run(["knowledge"], ["knowledge"], [], "nothing"),
    ]

    summary = ev.summarize(rows)

    assert summary["routing_accuracy"] == 0.5
    assert summary["mean_fact_coverage"] == 0.5
    assert summary["judge"]["correctness"] == 5 and summary["judged_cases"] == 1
    assert summary["latency_ms_p95"] == 10


def test_judge_rejects_malformed_scores(monkeypatch):
    monkeypatch.setattr(app, "ask_json", lambda *a, **k: {"correctness": 9, "groundedness": 4, "helpfulness": 3})
    assert ev.judge_case({**CASE, "message": "q"}, "a") is None

    monkeypatch.setattr(app, "ask_json", lambda *a, **k: {"correctness": 5, "groundedness": 4, "helpfulness": 3})
    assert ev.judge_case({**CASE, "message": "q"}, "a") == {
        "correctness": 5.0, "groundedness": 4.0, "helpfulness": 3.0}


def test_judge_agreement_against_human_scores():
    report = m.judge_agreement(
        {"a": {"correctness": 5}, "b": {"correctness": 2}},
        {"a": {"correctness": 4}, "b": {"correctness": 5}},
    )

    assert report["correctness"]["mean_abs_diff"] == 2.0
    assert report["correctness"]["within_one_point"] == 0.5


def test_scenario_passes_only_when_every_turn_has_its_facts(monkeypatch):
    answers = iter(["You have 320.00 declined: insufficient funds", "ok"])
    monkeypatch.setattr(app, "handle_traced",
                        lambda msg, uid, sid=None: ({"answer": next(answers), "session_id": "ses_12345678",
                                                     "agents_used": [], "escalated": False}, {}))
    scenario = {"id": "s", "turns": [
        {"message": "m1", "required_facts": ["320[.,]00", "insufficient"]},
        {"message": "m2", "required_facts": ["needs this"]},
    ]}

    outcome = ev.run_scenario(scenario)

    assert outcome["turns"][0]["fact_coverage"] == 1.0
    assert outcome["passed"] is False


def test_handoff_scenario_drives_the_operator_steps(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "claim_ticket", lambda t, o: calls.append(("claim", t, o)) or {})
    monkeypatch.setattr(app, "reply_to_ticket", lambda t, o, msg, key: calls.append(("reply", t, msg)) or {})
    answers = iter(["Registered (ticket TCK-ABC123).", "The specialist said 9am."])
    monkeypatch.setattr(app, "handle_traced",
                        lambda msg, uid, sid=None: ({"answer": next(answers), "session_id": "ses_12345678",
                                                     "agents_used": [], "escalated": True}, {}))
    scenario = {"id": "h", "turns": [
        {"message": "human please", "required_facts": ["TCK-"]},
        {"operator_reply": "Retry after 9am"},
        {"message": "any news?", "required_facts": ["9 ?am"]},
    ]}

    outcome = ev.run_scenario(scenario)

    assert calls == [("claim", "TCK-ABC123", "eval-operator"), ("reply", "TCK-ABC123", "Retry after 9am")]
    assert outcome["passed"] is True


def test_the_runner_refuses_to_run_without_a_key_and_writes_nothing(tmp_path):
    report = tmp_path / "r.json"
    env = {k: v for k, v in __import__("os").environ.items() if k != "GEMINI_API_KEY"}

    done = subprocess.run(
        [sys.executable, str(Path(ev.__file__)), "--report", str(report)],
        capture_output=True, text=True, env=env, cwd=ev.PROJECT_ROOT,
    )

    assert done.returncode == 2
    assert "GEMINI_API_KEY is not set" in done.stderr
    assert not report.exists()


def test_multiturn_dataset_is_well_formed():
    data = json.loads(ev.DEFAULT_MULTITURN.read_text(encoding="utf-8"))
    for scenario in data["scenarios"]:
        assert scenario["id"] and scenario["turns"]
        for turn in scenario["turns"]:
            assert ("message" in turn) != ("operator_reply" in turn)
