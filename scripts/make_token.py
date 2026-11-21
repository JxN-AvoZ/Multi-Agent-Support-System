"""Mint a short-lived HS256 token for local testing of AUTH_MODE=jwt.

    JWT_SECRET=<32+ chars> python scripts/make_token.py cliente1988

Prints the token. Use it as:  curl -H "Authorization: Bearer $TOKEN" ...
This script stands in for a real identity provider and is for development only.
"""

from __future__ import annotations

import os
import sys
import time

import jwt


def main() -> int:
    secret = os.getenv("JWT_SECRET", "")
    if len(secret) < 32:
        print("Set JWT_SECRET to at least 32 characters.", file=sys.stderr)
        return 1
    if len(sys.argv) != 2:
        print("usage: python scripts/make_token.py <customer_id>", file=sys.stderr)
        return 1

    now = int(time.time())
    claims = {"sub": sys.argv[1], "iat": now, "exp": now + 15 * 60}
    if os.getenv("JWT_ISSUER"):
        claims["iss"] = os.environ["JWT_ISSUER"]
    if os.getenv("JWT_AUDIENCE"):
        claims["aud"] = os.environ["JWT_AUDIENCE"]
    print(jwt.encode(claims, secret, algorithm="HS256"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
