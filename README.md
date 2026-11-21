# Getnet Multi-Agent Support System

A Python customer-support prototype for Getnet that combines a multi-agent workflow, customer-scoped support tools, a local Retrieval-Augmented Generation (RAG) knowledge base, persistent conversation memory, and a human-handoff loop.

Customer requests are routed to specialized agents instead of being handled by one general-purpose assistant. The project can be run, tested and partly evaluated **offline, with no Anthropic or Tavily key**. The model-backed workflow needs an Anthropic key; see [What has and has not been run](#what-has-and-has-not-been-run).

## Table of contents

- [Project objectives](#project-objectives)
- [Architecture](#architecture)
- [Main components](#main-components)
- [Video Overview](#video-overview)
- [RAG knowledge base](#rag-knowledge-base)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the API](#running-the-api)
- [Identity and authentication](#identity-and-authentication)
- [Conversation memory](#conversation-memory)
- [Human handoff](#human-handoff)
- [Observability](#observability)
- [Testing](#testing)
- [Evaluation](#evaluation)
- [What has and has not been run](#what-has-and-has-not-been-run)
- [Limitations and future improvements](#limitations-and-future-improvements)
- [Security considerations](#security-considerations)

## Project objectives

- Coordinate specialized agents to handle different customer-support requests.
- Separate account-specific operations from general product and policy questions.
- Retrieve information from Getnet's publicly available documentation and check answers against it.
- Provide customer-scoped tools for transaction, settlement, and terminal information.
- Keep conversation context across turns and hand unresolved cases to a human who can reply into the conversation.
- Expose the application through a REST API with a real authentication option.
- Evaluate retrieval offline and the full workflow end to end with a reproducible dataset.

## Architecture

```text
                          Customer request
                                 |
                                 v
              FastAPI  (identity: JWT or demo user_id)
                                 |
                       load session + history
                                 |
                                 v
                 Guardrails: injection check, redaction
                                 |
                                 v
                            Router Agent
                                 |
                      plan: support / knowledge / both
                                 |
                  +--------------+---------------+
                  |                              |
                  v                              v
            Support Agent                 Knowledge Agent
                  |                              |
      customer-scoped tools            query rewrite -> BM25 search
      (transactions, settlements,      -> grounded answer
       terminal, open ticket)          -> independent verifier
                  |                              |
                  +--------------+---------------+
                                 |
                  review: add the other agent? (at most once)
                                 |
                                 v
                  Combined response (compose step)
                                 |
                       Escalation if needed
                                 |
                                 v
        Ticket -> operator claims -> operator replies -> next turn
```

The router decides which specialists run and in what order. When both are needed, support runs first and its findings are passed to the knowledge agent. After a single-agent plan, a bounded review step can add the other agent once (see [Replanning](#replanning-and-verification)). The compose step merges the findings, and the escalation step decides whether a human should take over.

## Main components

### Video Overview

  Visit <https://youtu.be/I_iofOyIrio> for video description.

### 1. Router Agent

Classifies the request and returns a validated JSON plan: which agents to run and in which order, the customer's sentiment, whether they asked for a human, and whether the request is safe. If the `safe` field is missing the request is refused (fail closed). A model outage or empty payload returns a temporary-error message, not a policy refusal. The router also sees recent conversation history, so follow-ups such as "and the day before?" can be resolved.

### 2. Support Agent

Handles operations that depend on the current customer's data. Its tools are bound to the customer ID when the agent is built; the model cannot choose which account a tool reads.

| Tool | Purpose |
|---|---|
| `get_recent_transactions` | Recent transactions, optionally only declined ones. |
| `get_settlement_schedule` | Settlement and expected deposit information. |
| `get_terminal_status` | Terminal connection status and last error. |
| `open_ticket` | Create a persistent support ticket. |

Customer records are mock data. The tool loop is capped (default 6 rounds) and tool errors are passed back to the model flagged as errors.

### 3. Knowledge Agent

Answers product, procedure and policy questions from the local RAG index: an LLM rewrites the question into search queries, BM25-style retrieval finds passages, a model answers with numbered citations, and an **independent verifier** checks the answer against the same passages (see below). Tavily web search is used only when the question is outside Getnet documentation, or the local answer could not be grounded or verified.

### 4. Orchestration, replanning and escalation

`handle()` runs the selected agents in sequence, forwards findings between them, composes the answer, and decides on escalation. Escalation is triggered by: an explicit human request, frustrated or angry sentiment (configurable), a tool error, confidence below the threshold (0.45), or a ticket opened by the support agent. If the conversation already has an open case, it is reused instead of creating a duplicate.

#### Replanning and verification

- **Bounded replan.** After a plan that ran only one agent, one cheap model call reviews the result: if support found a problem whose fix needs documentation (for example an offline terminal), it adds the knowledge agent; if knowledge was unsure because the question depends on the customer's own data, it adds support. At most `MAX_REPLANS` (default 1) additions per request, an agent never runs twice, and any reviewer failure means "finish". It is not called when both agents already ran or when support's tools failed.
- **Independent verifier.** A separate model call (`VERIFY_MODEL`, default the fast model) audits each RAG answer against the passages. An unsupported answer is rejected and falls through to web search or an honest "could not find a reliable answer", which escalates. If the verifier itself is unavailable, the answer is kept but confidence is capped and `verified` stays unset. Disable with `VERIFY_GROUNDING=false`.

## RAG knowledge base

The knowledge base is built from publicly available Getnet website content:

1. Start from the configured seed URLs (Portuguese pages on `site.getnet.com.br`).
2. Fetch pages from allow-listed HTTPS hosts only, re-checking every redirect and refusing private IP addresses.
3. Extract text, split into 900-character chunks with 150 characters of overlap, and keep each chunk's source URL.
4. Store the chunks in a JSON index (written atomically; refreshed after 168 hours; a stale index is kept if a refresh fails).
5. Retrieve with a hand-written, accent-insensitive BM25-style ranking.

The repository ships a prebuilt `index.json` (242 chunks from 26 pages) so everything runs offline.

**Cross-language stopgap.** The pages are Portuguese and the search is lexical, so English questions retrieve poorly. A small static English-to-Portuguese term map expands query terms (`QUERY_EXPANSION`, on by default). It improves ranking measurably (see [Evaluation](#evaluation)) but it is a stopgap, not semantic search; embeddings or a reranker are the real fix. In production the LLM query rewrite covers the same gap more generally.

## Project structure

```text
.
├── main.py                     # API, agents, orchestration, crawler, retrieval, tools, memory, tickets
├── prompts.py                  # Agent instructions and response formats
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Runtime + test dependencies
├── Dockerfile                  # Container image (seeds the prebuilt index, runs as non-root)
├── docker-compose.yml          # API service and persistent data volume
├── index.json                  # Prebuilt RAG chunks (offline use)
├── pytest.ini                  # Lets pytest import main.py from the root
├── .env.example                # Every setting, all optional for the offline demo
├── scripts/
│   └── make_token.py           # Mints a development JWT for AUTH_MODE=jwt
├── tests/
│   ├── test_main.py            # Core behaviour: routing, tools, redaction, tickets, crawler
│   ├── test_identity.py        # JWT and demo identity
│   ├── test_memory.py          # Sessions, history, summary, retention, migration
│   ├── test_handoff.py         # Operator claim / reply / close, idempotency
│   ├── test_agentic.py         # Replanning and the groundedness verifier
│   ├── test_observability.py   # Token usage, logs, /metrics
│   ├── test_integration.py     # Whole stack over HTTP with a scripted stand-in model
│   ├── test_eval_runner.py     # Scoring logic of the evaluation runner
│   ├── eval_metrics.py         # Pure metric functions
│   ├── rag_eval.py             # Offline retrieval evaluation
│   └── agentic_eval.py         # End-to-end evaluation (needs an Anthropic key)
├── evaluation/
│   ├── rag_eval.json           # Dataset (dev and held-out cases)
│   ├── multiturn_cases.json    # Multi-turn and human-handoff scenarios
│   └── reports/rag_eval_report.json
└── .github/workflows/ci.yml    # Tests, retrieval regression gate, Docker build
```

## Requirements

- Python 3.12
- Docker and Docker Compose (optional; Compose 2.24+ for the optional `.env`)
- An Anthropic API key only for the model-backed workflow. A Tavily key is optional (web search).

## Installation

```bash
git clone https://github.com/JxN-AvoZ/Multi-Agent-Support-System.git
cd Multi-Agent-Support-System
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Running the API

### Offline (no keys)

```bash
uvicorn main:app --port 8000
```

Run this from the project root so the shipped `index.json` is used. The API starts, serves `/health`, applies guardrails and identity checks, and answers model-backed questions with a temporary-error message because no model is connected.

### With a model

```bash
cp .env.example .env            # then add GEMINI_API_KEY (and optionally TAVILY_API_KEY)
set -a; source .env; set +a     # the app reads environment variables, not the file
uvicorn main:app --port 8000
```

### Docker

```bash
docker compose up --build       # works offline; add .env for keys
docker compose run --rm api python -m pytest -q
```

The image seeds the prebuilt index into the `/data` volume, so it starts without crawling.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Status and number of indexed chunks. |
| `GET` | `/metrics` | In-process counters and latency percentiles (no customer data). |
| `POST` | `/chat` | Process a customer request. Body: `{"message", "user_id"?, "session_id"?}`. |
| `GET` | `/tickets/{id}` | The customer's own ticket: status and the operator's latest reply. |
| `GET` | `/operator/tickets` | Operator queue (`?status=open`). |
| `GET` | `/operator/tickets/{id}` | One ticket and its conversation. |
| `POST` | `/operator/tickets/{id}/claim` | Claim an open ticket. |
| `POST` | `/operator/tickets/{id}/reply` | Reply into the customer's conversation (idempotent). |
| `POST` | `/operator/tickets/{id}/close` | Close a ticket. |

The response always has the same shape: `answer`, `agents_used`, `sources`, `confidence`, `escalated`, `trace_id`, `latency_ms` and `session_id`.

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"My card machine will not connect to the internet","user_id":"cliente1988"}'
```

Interactive documentation: `http://localhost:8000/docs`.

## Identity and authentication

`AUTH_MODE` selects how the customer is identified:

| Mode | Customer comes from | Use |
|---|---|---|
| `demo` (default) | the `user_id` in the request body | The challenge payload. **Not authentication**: anyone who knows an ID can query it. |
| `jwt` | the `sub` claim of a verified HS256 bearer token | A real boundary. A body `user_id` is optional and must match the token, so no request-body value can choose the customer. |

In `jwt` mode: tokens need `exp` and `sub`; only HS256 is accepted; `JWT_ISSUER` and `JWT_AUDIENCE` are enforced when set; `JWT_SECRET` must be at least 32 characters or startup fails. Wrong signature, expiry, algorithm or issuer gives 401; a mismatching `user_id` or an unknown customer gives 403.

```bash
export AUTH_MODE=jwt JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
uvicorn main:app --port 8000 &
TOKEN=$(python scripts/make_token.py cliente1988)     # development stand-in for an identity provider
curl -s localhost:8000/chat -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"message":"Show my recent transactions"}'
```

`demo` is the default so the challenge payload `{"message", "user_id"}` works unchanged; the server logs a warning at startup in that mode. For anything real, use `jwt` with an identity provider (preferably asymmetric keys / JWKS rather than a shared HS256 secret). Everything downstream receives only the resolved customer ID, and tools are bound to it.

## Conversation memory

Every response carries a `session_id`. Send it back to continue the conversation.

- The last `MEMORY_MESSAGES` (8) turns are passed to the router, support agent, knowledge agent and compose step as escaped, tagged, **untrusted** history.
- When a session grows beyond `SUMMARY_TRIGGER_MESSAGES` (14), older turns are folded into a short rolling summary by a cheap model call. If summarising fails, the reply is unaffected.
- A session belongs to the customer who created it; another customer's token gets 403.
- Stored text is redacted (cards, CPFs). Refused and failed turns are not stored, so an injection attempt cannot return later as history. Sessions are purged after `CONVERSATION_TTL_DAYS` (30).
- Storage failure degrades to a stateless reply. Memory shares the SQLite file with tickets (`TICKET_DB_FILE`); existing ticket databases are migrated in place.

Limit: the summary is model-written from customer text, so it is treated as untrusted data like the rest of the history; it reduces but does not remove the risk of stored prompt injection.

## Human handoff

Escalation creates a ticket tied to the conversation, and a human can answer into it:

```text
customer -> agent -> ticket (open) -> operator claims (claimed) -> operator replies (replied)
        -> the reply is stored in the session -> the customer's next turn sees it and the agent resumes
```

- Operators use the `/operator/*` routes with an `X-Operator-Key` header. They are **off** unless `OPERATOR_API_KEY` (24+ characters) is set.
- Claiming is atomic: of two operators racing for one ticket, one wins and the other gets 409. Only the claiming operator can reply. Closed tickets take no more replies.
- Replies accept an `idempotency_key`; retrying with the same key never stores the reply twice. Operator text is redacted before the customer sees it.
- The agent resumes on the customer's **next message**; there is no push channel to the customer.
- Limit: a single shared operator key and a self-declared `operator_id` are a prototype. Production needs per-operator identity (SSO), an assignment queue and audit logging.

## Observability

- **Logs:** one JSON object per line, with a `trace_id` on every event. Message text and customer IDs are never logged, only lengths and metadata. Each agent step logs its latency; each request logs token usage (`input_tokens`, `output_tokens`, `model_calls`).
- **Cost:** estimated only if you provide prices in `MODEL_PRICES_USD_PER_MTOK`; the app never guesses prices.
- **`GET /metrics`:** request count, escalation / block / router-failure rates, applied replans, answers that failed verification, token totals, and p50 / p95 latency over the last 1,000 requests. Counters are per process; use a real metrics backend when running several replicas.

## Testing

```bash
python -m pytest -q
```

About 100 tests, all offline, with the model and network replaced by stand-ins. They cover orchestration, prompt-injection handling, router validation and failure modes, customer-scoped tools, redaction, tickets and migration, JWT and demo identity, memory, the operator loop, replanning, the verifier, observability, the evaluation runner's scoring, and one **whole-stack test over HTTP** (`tests/test_integration.py`) that drives JWT, router, support tool loop, knowledge, verifier, compose, memory, escalation, operator reply and metrics together against a scripted model.

Stand-ins show that our code is correct when the model returns well-formed output. They cannot show that the real model behaves; that is what `tests/agentic_eval.py` is for.

## Evaluation

### Offline retrieval evaluation

```bash
python tests/rag_eval.py --top-k 5            # no keys, no network
QUERY_EXPANSION=false python tests/rag_eval.py --top-k 5 --report /tmp/off.json   # ablation
```

The dataset (`evaluation/rag_eval.json`) has 21 cases (17 answerable from the local index), split into **dev** (15 cases) and **held-out** (6 cases added to check generalisation). The runner reports diagnostic metrics (non-empty retrieval, keyword coverage) and ranking metrics against page-level relevance labels: Precision@K, Recall@K, MRR and Hit@K.

Current results (K = 5, 242 chunks, shipped `index.json`):

| Setting | Non-empty | Keyword coverage | P@5 | R@5 | MRR | Hit@5 |
|---|---:|---:|---:|---:|---:|---:|
| Query expansion on (default) | 16/17 | 64.7% | 0.69 | 0.76 | 0.78 | 0.94 |
| Query expansion off | 15/17 | 64.7% | 0.69 | 0.67 | 0.73 | 0.81 |
| On, dev split (10 labeled) | | | 0.68 | 0.72 | 0.88 | 1.00 |
| On, held-out split (6 labeled) | | | 0.72 | 0.83 | 0.62 | 0.83 |

How to read this honestly:

- Labels are **page-level** (a URL substring) and were written by the project author, not an independent annotator. They measure "the right page was retrieved", not "the right sentence".
- The held-out set is small (6 cases), so differences there are noisy. The term map and the labels were developed alongside the dev cases, so read the dev numbers as optimistic and the held-out numbers as the fairer (but noisier) signal.
- Expansion helps recall and rank but does not change keyword coverage; one question (declined-transaction errors) still retrieves no relevant page. Lexical search has a ceiling; embeddings or a reranker are the next step and should be judged on this same dataset.
- The original submission measured 11 cases (10/11 non-empty, 52.3% coverage). The dataset has since grown, so those numbers are not directly comparable.

The CI job fails the build if Hit@5 drops below 0.8 or MRR below 0.6 (`--min-hit-rate`, `--min-mrr`).

### End-to-end agentic evaluation (needs a key)

```bash
GEMINI_API_KEY=... python tests/agentic_eval.py --judge
python tests/agentic_eval.py --write-human-template     # blank sheet for hand scoring
GEMINI_API_KEY=... python tests/agentic_eval.py --judge --human-labels evaluation/human_labels.json
```

This runs the real workflow for every case and reports routing accuracy (the router's plan, and the agents that finally ran after any replan), tool-selection precision and recall, required-fact coverage (regexes over the answer), verifier outcomes, escalation rate, latency and tokens, plus optional **LLM-judge** scores (correctness, groundedness, helpfulness, 1 to 5) and the multi-turn / human-handoff scenarios in `evaluation/multiturn_cases.json`. It uses a temporary database, skips web-search cases when `TAVILY_API_KEY` is unset, and refuses to run (exit code 2, no report) without an Anthropic key.

The judge is a model grading a model. Before trusting its averages, hand-score a sample and pass it with `--human-labels`; the report then includes the judge's mean absolute difference and within-one-point rate against the human scores. **No end-to-end numbers are published here yet because I have not run it against a live model.**

## What has and has not been run

| Verified in this repository | Not run end to end |
|---|---|
| API contract, validation, identity (both modes), guardrails, tool scoping, ticket store and migration, memory, operator loop, metrics | Router, query rewrite, tool-use loop, verifier, replan review and answer generation against a **live model** (covered by stand-ins and a scripted whole-stack test) |
| Retrieval over the real index and its ranking metrics | The crawler against the live Getnet site in this environment |
| 100+ offline tests, CI configuration | `docker build` in the environment where this was written (the CI workflow builds and tests the image) |

## Limitations and future improvements

- **Live model path unmeasured.** Run `tests/agentic_eval.py` and publish the numbers; the harness, labels and scenarios are ready.
- **Retrieval is lexical.** Cross-language and vocabulary gaps remain; add embeddings or hybrid retrieval with a reranker and judge it on the same labeled set. Relevance labels are page-level and author-written.
- **Replanning is bounded, not open-ended.** One review step after a single-agent plan; no iterative planner or critic loop.
- **The verifier shares a model family with the answerer** (a cheaper tier, different prompt). It lowers the risk of unsupported answers but is not a guarantee.
- **Mock data.** Replace `users()` with authenticated Getnet services.
- **Identity.** `jwt` mode is real but uses a shared HS256 secret; production needs an identity provider (JWKS) and an authorization layer. `demo` mode is not authentication.
- **Human handoff** has no push channel, no per-operator identity and no SLA or queue assignment.
- **Single host.** SQLite and in-process metrics suit one instance; use a shared database and metrics backend for several.
- **Confidence is a heuristic** and the injection check is pattern-based; neither is a security boundary on its own.

## Security considerations

This is a technical prototype, not a production-ready payment-support system.

- Customer records are mock data. Tools are bound to the resolved customer ID; the model cannot select an account.
- Identity: use `AUTH_MODE=jwt` outside demos. Secrets belong in environment variables or a local `.env` that is never committed (`.gitignore` excludes it; `.env.example` is committed).
- Prompt-injection defences are layered: a pattern check before any model call, a fail-closed router safety flag, escaping and tagging of all untrusted text (history, retrieved pages, agent findings), validated model JSON, and customer-scoped tools. None of these is claimed to be complete.
- The crawler allows only configured HTTPS hosts on the standard port, re-validates redirects and DNS, and limits page count, bytes and text.
- Valid payment-card numbers (Luhn-checked) and formatted CPFs are redacted in input, output, stored history and tickets.
- Operator routes are disabled unless a key is configured and use constant-time comparison.

---

## License

 GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007
