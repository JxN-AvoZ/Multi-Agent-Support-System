"""Identity tests: the customer comes from a verified token, not the body."""

from __future__ import annotations

import time

import jwt
import pytest
from fastapi.testclient import TestClient

import main as app

SECRET = "s" * 48


@pytest.fixture(autouse=True)
def isolate_ticket_database(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "TICKET_DB_FILE", tmp_path / "test-tickets.sqlite3")


@pytest.fixture
def jwt_mode(monkeypatch):
    monkeypatch.setattr(app, "AUTH_MODE", "jwt")
    monkeypatch.setattr(app, "JWT_SECRET", SECRET)
    monkeypatch.setattr(app, "JWT_ISSUER", None)
    monkeypatch.setattr(app, "JWT_AUDIENCE", None)
    monkeypatch.setattr(app, "handle", lambda message, uid, **_: {"answer": "ok", "uid": uid})


@pytest.fixture
def client():
    return TestClient(app.app)


def token(sub="cliente1988", secret=SECRET, algorithm="HS256", exp_in=300, **extra):
    claims = {"sub": sub, **extra}
    if exp_in is not None:
        claims["exp"] = int(time.time()) + exp_in
    return jwt.encode(claims, secret, algorithm=algorithm)


def post(client, body, tok=None):
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    return client.post("/chat", json=body, headers=headers)


def test_token_alone_identifies_the_customer(jwt_mode, client):
    response = post(client, {"message": "hi"}, token())

    assert response.status_code == 200
    assert response.json()["uid"] == "cliente1988"


def test_matching_body_user_id_is_accepted(jwt_mode, client):
    response = post(client, {"message": "hi", "user_id": "cliente1988"}, token())

    assert response.status_code == 200


def test_body_user_id_cannot_override_the_token(jwt_mode, client):
    response = post(client, {"message": "hi", "user_id": "someone-else"}, token())

    assert response.status_code == 403


def test_missing_token_is_rejected_even_with_a_valid_user_id(jwt_mode, client):
    response = post(client, {"message": "hi", "user_id": "cliente1988"})

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "bad",
    [
        lambda: token(exp_in=-10),  # expired
        lambda: token(secret="x" * 48),  # wrong signing secret
        lambda: token(algorithm="HS512"),  # algorithm not on the allow-list
        lambda: token(exp_in=None),  # no expiry claim
        lambda: "not-a-jwt",
    ],
)
def test_invalid_tokens_are_rejected(jwt_mode, client, bad):
    assert post(client, {"message": "hi"}, bad()).status_code == 401


def test_token_for_an_unknown_customer_is_forbidden(jwt_mode, client):
    assert post(client, {"message": "hi"}, token(sub="nobody")).status_code == 403


def test_weak_secret_fails_closed(monkeypatch, jwt_mode, client):
    monkeypatch.setattr(app, "JWT_SECRET", "short")

    assert post(client, {"message": "hi"}, token(secret="short")).status_code == 503


def test_issuer_and_audience_are_enforced_when_configured(monkeypatch, jwt_mode, client):
    monkeypatch.setattr(app, "JWT_ISSUER", "idp.example")
    monkeypatch.setattr(app, "JWT_AUDIENCE", "getnet-support")

    good = token(iss="idp.example", aud="getnet-support")
    wrong_issuer = token(iss="evil.example", aud="getnet-support")
    no_audience = token(iss="idp.example")

    assert post(client, {"message": "hi"}, good).status_code == 200
    assert post(client, {"message": "hi"}, wrong_issuer).status_code == 401
    assert post(client, {"message": "hi"}, no_audience).status_code == 401


def test_demo_mode_still_accepts_the_challenge_payload(monkeypatch, client):
    monkeypatch.setattr(app, "AUTH_MODE", "demo")
    monkeypatch.setattr(app, "handle", lambda message, uid, **_: {"answer": "ok", "uid": uid})

    ok = client.post("/chat", json={"message": "hi", "user_id": "cliente1988"})
    missing = client.post("/chat", json={"message": "hi"})

    assert ok.status_code == 200 and ok.json()["uid"] == "cliente1988"
    assert missing.status_code == 422


def test_startup_refuses_jwt_mode_without_a_strong_secret(monkeypatch):
    monkeypatch.setattr(app, "AUTH_MODE", "jwt")
    monkeypatch.setattr(app, "JWT_SECRET", None)

    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        with TestClient(app.app):
            pass
