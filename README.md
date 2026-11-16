# Getnet Multi-Agent Support System

A Python-based customer-support prototype for Getnet that combines a multi-agent workflow, customer-scoped support tools, and a local Retrieval-Augmented Generation (RAG) knowledge base.

The project is designed to demonstrate how customer requests can be routed to specialized agents instead of being handled by one general-purpose assistant. It also includes an offline evaluation script for measuring the retrieval quality of the local knowledge base without making paid Anthropic or Tavily API calls.

## Table of contents

- [Project objectives](#project-objectives)
- [Architecture](#architecture)
- [Main components](#main-components)
- [RAG knowledge base](#rag-knowledge-base)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the API](#running-the-api)
- [Testing](#testing)
- [Offline RAG evaluation](#offline-rag-evaluation)
- [Evaluation results](#evaluation-results)
- [Limitations and future improvements](#limitations-and-future-improvements)
- [Security considerations](#security-considerations)

## Project objectives

The main objectives are to:

- Coordinate specialized agents to handle different customer-support requests.
- Separate account-specific operations from general product and policy questions.
- Retrieve information from Getnet's publicly available documentation.
- Provide customer-scoped tools for transaction, settlement, and terminal information.
- Escalate unresolved requests through persistent support tickets.
- Expose the application through a REST API.
- Evaluate local document retrieval using a reproducible JSON test dataset.

## Architecture

The application follows a router-and-specialists pattern:

```text
                         Customer request
                                |
                                v
                         FastAPI endpoint
                                |
                                v
                         Input validation
                                |
                                v
                           Router Agent
                                |
                  Selects the execution plan
                                |
                  +-------------+-------------+
                  |                           |
                  v                           v
             Support Agent              Knowledge Agent
                  |                           |
          Customer-scoped tools          Local RAG index
                  |                           |
          +-------+--------+             BM25-style search
          |       |        |                  |
       Recent  Settlement Terminal       Retrieved chunks
       sales   schedule   status               |
          |       |        |                   |
          +-------+--------+-------------------+
                                |
                                v
                      Combined response
                                |
                     Escalation if needed
                                |
                                v
                       Support ticket
```

The Router Agent determines which specialists should run and in which order. When both specialists are needed, the Support Agent runs first so its findings can be passed to the Knowledge Agent. The orchestration layer then combines the available findings into a customer-facing response.

## Main components

### 1. Router Agent

The router classifies the customer's request and produces a structured execution plan. It identifies whether the request requires:

- Customer-specific support data.
- General Getnet product or policy knowledge.
- Both support and knowledge specialists.

It also classifies customer sentiment, detects explicit requests for human assistance, and marks unsafe requests for refusal.

### 2. Support Agent

The Support Agent handles operations that depend on the current customer's information. Its tools include:

| Tool | Purpose |
|---|---|
| `get_recent_transactions` | Retrieve recent transactions, including declined transactions when requested. |
| `get_settlement_schedule` | Retrieve settlement and expected deposit information. |
| `get_terminal_status` | Check the customer's payment-terminal status and connection information. |
| `open_ticket` | Create a persistent support ticket when further assistance is required. |

The current customer records are mock data intended for demonstration and testing. Support tickets are stored in SQLite.

### 3. Knowledge Agent

The Knowledge Agent answers questions about Getnet products, services, procedures, and policies. It uses the local RAG index to retrieve relevant passages from crawled Getnet pages.

The retrieval implementation uses a BM25-style lexical ranking approach. Retrieved chunks retain their source URLs so the application can trace information back to the original pages.

### 4. Orchestration and escalation

The `handle()` function coordinates the workflow. It executes the selected agents in sequence, forwards findings between specialists, combines their answers, and evaluates whether escalation is necessary.

Escalation can be triggered by conditions such as an explicit human request, frustrated or angry sentiment (depending on configuration), tool errors, low confidence, or an existing ticket created by a support tool.

## RAG knowledge base

The local knowledge base is created from publicly available Getnet website content.

The ingestion and retrieval workflow is:

1. Start from the configured seed URLs.
2. Fetch pages from explicitly allowed HTTPS hosts.
3. Extract textual content from the HTML.
4. Divide the extracted text into overlapping chunks.
5. Store each chunk with its source URL in a local JSON index.
6. Build retrieval statistics from the indexed chunks.
7. Search the index and return the top-K results for a query.

The crawler includes host and redirect validation, as well as limits on page count, download size, and extracted text size. The index is persisted locally and can be reused between runs.

**Retrieval note:** BM25-style lexical retrieval is based on term matching and ranking. It is not a semantic embedding search. As a result, vocabulary differences and cross-language questions (for example, English questions against Portuguese documentation) can affect retrieval quality.

## Project structure

```text
Multiagent Support System/
├── main.py                         # API, agents, orchestration, crawler, retrieval and tools
├── prompts.py                      # Agent instructions and response formats
├── requirements.txt                # Runtime dependencies
├── requirements-dev.txt            # Runtime and development/test dependencies
├── Dockerfile                      # Container image definition
├── docker-compose.yml              # API service and persistent data volume
├── index.json                      # Locally persisted RAG chunks
├── tests/
│   ├── test_main.py                # Automated tests
│   └── rag_eval.py                 # Offline RAG evaluation runner
└── evaluation/
    ├── rag_eval.json               # Evaluation dataset
    └── reports/
        └── rag_eval_report.json    # Generated evaluation report
```

Runtime files such as the SQLite ticket database may be created when the application runs. Their locations can be configured through environment variables.

## Requirements

- Python 3.12
- Conda (optional, for environment management)
- Git (optional)
- Docker and Docker Compose (optional, for container execution)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create the Conda environment

```bash
conda create -n getnet python=3.12 -y
conda activate getnet
```

### 3. Install dependencies

For development and testing:

```bash
pip install -r requirements-dev.txt
```

The development requirements file installs the runtime dependencies as well as pytest and httpx.

## Running the API

The API is served with Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Available endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Returns application status and the number of indexed chunks. |
| `POST` | `/chat` | Processes a customer-support request. |

Example health check:

```bash
curl http://localhost:8000/health
```

Example chat request:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"How can I connect my Getnet terminal to Wi-Fi?","user_id":"cliente1988"}'
```

Interactive API documentation is available at:

```text
http://localhost:8000/docs
```

**API integration note:** The complete live multi-agent `/chat` workflow uses the Anthropic client for model-driven routing and agent behavior. Configure a valid `ANTHROPIC_API_KEY` in your local environment before using that workflow. The offline RAG evaluation described below does not call Anthropic or Tavily.

## Testing

Run the automated test suite from the project root:

```bash
pytest -v
```

The tests cover application behavior such as:

- Agent orchestration and forwarding findings between agents.
- Prompt-injection handling.
- Router output validation and failure handling.
- Customer-scoped tool access.
- Transaction filtering and ordering.
- Card-number and CPF redaction.
- Ticket creation and persistence.
- API request validation and health reporting.

The test suite uses mocks and local test data where appropriate, so these tests do not require paid model API calls.

## Offline RAG evaluation

The project includes a dedicated evaluation runner at `tests/rag_eval.py` and a JSON dataset at `evaluation/rag_eval.json`.

Run the evaluation from the project root:

```bash
python tests/rag_eval.py \
  --dataset evaluation/rag_eval.json \
  --top-k 5
```

To display a preview of retrieved text:

```bash
python tests/rag_eval.py \
  --dataset evaluation/rag_eval.json \
  --top-k 5 \
  --show-text
```

The script:

1. Loads and validates the JSON evaluation dataset.
2. Selects cases configured for RAG evaluation.
3. Loads the local index and initializes retrieval statistics.
4. Runs the local search function for each evaluation question.
5. Checks retrieved content against expected keywords and source hints.
6. Calculates diagnostic retrieval metrics.
7. Saves a JSON report to `evaluation/reports/rag_eval_report.json` by default.

The evaluator supports an optional `retrieval_queries` array in each case. This allows manually supplied query variants (for example, Portuguese search terms for an English question) to be tested without an LLM-based query-rewriting call.

### Offline evaluation and external services

The RAG evaluation runner calls the local retrieval functions directly. It does not call Anthropic or Tavily and does not generate LLM answers. Therefore, it can be used without purchasing API credits, provided that a local index is already available.

If the index is missing or needs refreshing, `main.load_index()` may run the crawler against the configured Getnet seed URLs. In that situation, network access to the allowed source website is needed to build or refresh the local index. Once the index is available, retrieval evaluation can run against the saved local data.

Anthropic-powered query rewriting, answer generation, and dynamic agent orchestration, as well as Tavily-powered external search, are separate capabilities of the broader application and can be considered when extending the offline evaluation workflow in the future.

### Evaluation metrics

The current evaluator reports:

- **Non-empty retrieval rate:** the proportion of evaluated questions for which at least one chunk was retrieved.
- **Average keyword coverage:** the average proportion of expected keywords found in the retrieved content.
- **Source-hint overlap:** a lexical overlap check between natural-language source hints and retrieved text.

These are diagnostic metrics. They do not, by themselves, establish answer accuracy or formal Precision@K and Recall@K. Source relevance should also be reviewed manually. Formal ranking metrics would require explicit relevance labels or document identifiers in the dataset.

## Evaluation results

The latest report included in this project was generated with the following configuration:

| Metric | Result |
|---|---:|
| Dataset version | 2.0 |
| Total dataset cases | 15 |
| RAG cases evaluated | 11 |
| Non-RAG cases skipped | 4 |
| Indexed chunks | 242 |
| Top-K | 5 |
| Non-empty retrieval | 10/11 (90.9%) |
| Average keyword coverage | 52.3% |

The results show that the local index returned at least one chunk for most RAG questions. However, keyword coverage indicates room for improvement, particularly for cross-language queries and topics where relevant terms are not consistently represented in the retrieved passages.

A non-empty retrieval result should not automatically be interpreted as a relevant result. Likewise, keyword coverage is a lexical diagnostic rather than a semantic evaluation of answer quality.

## Limitations and future improvements

The current implementation is a prototype and has several areas that could be extended:

- **Semantic retrieval:** add embeddings or hybrid retrieval to improve matching when questions and documents use different vocabulary.
- **Query rewriting:** use an LLM to generate multiple targeted retrieval queries dynamically.
- **External search:** integrate Tavily for general questions that cannot be answered from Getnet documentation, such as current weather or exchange rates.
- **Answer generation:** use an LLM to generate grounded natural-language responses from retrieved passages.
- **Retrieval evaluation:** add explicit relevant-document labels and ranking metrics such as Precision@K, Recall@K, and Mean Reciprocal Rank.
- **Production integrations:** replace mock customer data with authenticated Getnet services and replace single-host SQLite ticket storage with an appropriate shared persistence service.
- **Authentication:** derive customer identity from a verified authentication token rather than trusting a user ID supplied in the request body.

Anthropic and Tavily are not dependencies of the offline RAG evaluation process. Their use for expanded model-driven orchestration, answer generation, query rewriting, and external search can be treated as a future extension of the evaluation setup.

## Security considerations

This project is a technical prototype, not a production-ready payment-support system.

- Customer records are mock data.
- The request's `user_id` is a demonstration identity, not proof of authentication. The code comments describe the need to derive customer identity from a verified token in production.
- API keys and other secrets must be kept in a local `.env` file or environment variables and must never be committed to version control.
- The crawler restricts access to configured HTTPS hosts and validates redirects.
- Sensitive values such as valid payment-card numbers and CPF identifiers are redacted by the application.
- Production deployment would require additional security, privacy, monitoring, authentication, and operational controls.

---

## License

No license has been specified for this project yet.
