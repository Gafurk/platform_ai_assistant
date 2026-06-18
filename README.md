# iSEL AI Assistant

FastAPI RAG chatbot for the **iSEL** platform. Answers user questions in **Russian** and **Kazakh** using a LightRAG knowledge graph (graph + vector hybrid) backed by OpenAI.

- **UI:** `http://localhost:8001/static/index.html`
- **Health:** `GET /health` → `{"status": "ok", "service": "isel-bot"}`

---

## Features

- Answers in **Russian and Kazakh** with automatic language detection and Kazakh transliteration
- **14 service intents** — each maps to a guided step-by-step flow (LinearFlow, ScenarioFlow, FAQFlow)
- **LightRAG hybrid retrieval** — graph traversal + vector similarity, no hallucinated navigation
- **Entity gate** — blocks ФЛ/ЮЛ-specific steps until the user identifies their entity type
- **FAQ interruption** — answers side questions mid-flow without losing flow state
- **Rule-based layer** — greetings, identity, navigation handled with zero LLM calls
- **Prefix-cached prompts** — static system prompt reused across requests (OpenAI cache hit)
- **Document ingestion** — PDF, DOCX, TXT, MD → structured situation/step chunks → LightRAG KG
- **Streaming reindex** — full KG rebuild via SSE with per-file progress
- **Docker-ready** — multi-stage build, non-root user, embedded KG (no external DB)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  POST /api/v1/chat                                              │
│                                                                 │
│  validate → rate_limit → rule_based ──────────────► response   │
│       │                                                         │
│  KZ transliteration → language detect → intent classify        │
│       │                                                         │
│  navigation shortcut ─────────────────────────────► LLM + RAG  │
│  FAQ interruption gate ────────────────────────────► LLM + RAG  │
│       │                                                         │
│  entity detect → entity clarification gate                     │
│       │                                                         │
│  flow.next_state()  (LinearFlow / ScenarioFlow / FAQFlow)      │
│       │                                                         │
│  search_docs() ── LightRAG hybrid ── intent/step filter        │
│       │                                                         │
│  ask_llm() ── static cached prompt + dynamic suffix            │
│       │                                                         │
│  session.update() ─────────────────────────────────► response   │
└─────────────────────────────────────────────────────────────────┘

LightRAG KG (embedded)
  ├── Graph store: NetworkX   ─┐
  └── Vector DB: nano-vectordb ┘  persisted to data/lightrag/
```

---

## Requirements

- Python 3.11+
- OpenAI API key (`OPENAI_API_KEY`)
- Docker + Docker Compose (for containerised deployment)

---

## Quick Start

### Option A — Docker (recommended)

```bash
# 1. Configure
cp .env.example .env      # set OPENAI_API_KEY inside

# 2. Start
docker compose up --build

# 3. Ingest documents (first run)
docker compose exec api python ingestion/ingest.py

# 4. Open the UI
# http://localhost:8001/static/index.html
```

`./data` (KG + uploaded docs) and `./logs` persist across restarts via bind mounts.

### Option B — Local (venv)

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # PowerShell
# source venv/Scripts/activate   # bash / Git Bash

pip install -r requirements.txt
cp .env.example .env             # set OPENAI_API_KEY

python ingestion/ingest.py       # ingest documents first
uvicorn app.main:app --reload --port 8001
```

---

## Configuration

All configuration is via environment variables. Copy `.env.example` → `.env` and fill in at minimum `OPENAI_API_KEY`.

| Variable | Required | Default | Notes |
|---|---|---|---|
| `OPENAI_API_KEY` | **Yes** | — | |
| `LLM_PROVIDER` | No | `openai` | `openai` or `ollama` |
| `OPENAI_MODEL` | No | `gpt-4.1-nano` | Chat responses — reasoning models supported (e.g. `gpt-5-mini`) |
| `LIGHTRAG_EXTRACT_MODEL` | No | `gpt-4.1-nano` | Entity extraction during ingestion — **non-reasoning only** |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama fallback |
| `OLLAMA_MODEL` | No | `llama3.2` | Ollama fallback |

> **`LIGHTRAG_EXTRACT_MODEL` must be a non-reasoning model.** Reasoning models (`o`-series, `gpt-5-mini`) exhaust `max_completion_tokens` on LightRAG's 14 KB extraction prompt and produce 0 entities. `OPENAI_MODEL` (chat) can be any model.

Service intents and navigation paths are configured in `config/services.yaml`. Intent keywords live in `config/keywords.yaml`. **Adding a new service requires only YAML edits — no code changes.**

---

## Data & Ingestion

```
data/
├── docs/        ← place source documents here (.pdf .docx .txt .md)
└── lightrag/    ← persistent KG (graph + embeddings) — BACK THIS UP
```

Neither directory is committed. `data/lightrag/` is created automatically on first run.

**Ingest all documents:**
```bash
python ingestion/ingest.py                          # local
docker compose exec api python ingestion/ingest.py  # Docker
```

**Upload single file + trigger reindex via API:**
```bash
curl -X POST http://localhost:8001/api/v1/upload -F "file=@myfile.pdf"
curl -X POST http://localhost:8001/api/v1/reindex
```

Ingestion cost: ~0.5–2 USD per 100 KB document (entity extraction via OpenAI).

---

## API

### `POST /api/v1/chat`
```json
// Request
{ "message": "Как подать заявку на ТУ?", "session_id": "user-abc", "page": "шаг 1" }

// Response
{ "answer": "...", "source": "llm", "handoff": false }
```
`source`: `"rule_based"` | `"llm"` | `"validation"` | `"rate_limit"`

### `GET /health`
```json
{"status": "ok", "service": "isel-bot"}
```

### `POST /api/v1/upload`
Multipart. Saves to `data/docs/`. Formats: `.pdf` `.docx` `.txt` `.md`

### `GET /api/v1/documents`
Lists files in `data/docs/`.

### `DELETE /api/v1/documents/{filename}`
Removes file from disk only — KG nodes persist until reindex.

### `POST /api/v1/reindex`
Full wipe of `data/lightrag/` + rebuild. Streams SSE progress:
```
data: {"status": "wiping"}
data: {"status": "progress", "file": "foo.pdf", "done": 1, "total": 5}
data: {"status": "done", "total": 5}
```

---

## Frontend

Static glassmorphism chat widget served at `http://localhost:8001/static/index.html`.

Single-file (`frontend/index.html`) — vanilla HTML/CSS/JS, no build step required. The `frontend/` directory must exist before server start (FastAPI `StaticFiles` raises on missing directory).

---

## Testing

```bash
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=app --cov-report=term-missing
```

200+ tests covering flows, prompt builder, service registry, session, transliteration, rule-based layer.

---

## Project Structure

```
app/
├── config/service_registry.py   # Singleton — loads YAML once at import
├── flows/
│   ├── base.py                  # FlowContext + BaseFlow interface
│   ├── linear.py                # Step-by-step flows (entity-gated where needed)
│   ├── scenario.py              # Branching flows (real estate)
│   ├── faq.py                   # Pure retrieval, no state
│   └── registry.py              # get_flow(), classify_intent(), INTENT_KEYWORDS
├── models/schemas.py            # ChatRequest, ChatResponse, FlowState
├── routers/
│   ├── chat.py                  # POST /api/v1/chat
│   └── documents.py             # upload / list / delete / reindex
├── services/
│   ├── lightrag_service.py      # LightRAG init, search_docs(), insert_text()
│   ├── rag.py                   # Thin wrapper re-exporting search_docs()
│   ├── llm_service.py           # OpenAI / Ollama call
│   ├── prompt_builder.py        # Static base prompt + dynamic suffix
│   ├── rulebased.py             # SYSTEM_COMMANDS + NAVIGATION_RULES (zero LLM)
│   ├── translit.py              # Kazakh transliteration (Russian letters → KZ Cyrillic)
│   ├── session.py               # SessionManager (history capped at 10 entries)
│   ├── rate_limit.py            # Per-session rate limiter
│   └── validation.py            # Input validation
└── utils/logger.py              # Structured logging

config/
├── services.yaml                # 14 service intents, flow types, nav paths
└── keywords.yaml                # Intent keywords

ingestion/ingest.py              # Document parser + LightRAG inserter

frontend/index.html              # Static chat widget (glassmorphism)
tests/                           # pytest suite
Dockerfile                       # Multi-stage build (python:3.11-slim, non-root)
docker-compose.yml               # Single api service; mounts ./data + ./logs
```

---

## Supported Services (14 intents)

| Intent | Display name (RU) | Flow | Steps | Entity gate |
|---|---|---|---|---|
| `tu_application` | Заявление на технические условия | Linear | 5 | Yes |
| `primary_connection_residential` | Первичное подключение (бытовой) | Linear | 4 | No |
| `primary_connection_nonresidential` | Первичное подключение (небытовой) | Linear | 5 | No |
| `secondary_connection` | Вторичное подключение | Linear | 3 | Yes |
| `contract_termination` | Расторжение договора | Linear | 3 | Yes |
| `grid_disconnection` | Отключение от электросетей | Linear | 2 | Yes |
| `equipment_testing` | Испытание, измерение электрооборудования | Linear | 4 | Yes |
| `real_estate` | Добавление объекта недвижимости | Scenario | — | No |
| `supply_contract_residential` | Договор электроснабжения (бытовой) | Linear | 4 | No |
| `supply_contract_non_residential` | Договор электроснабжения (небытовой) | Linear | 4 | No |
| `load_calculation` | Расчёт электрической нагрузки | Linear | 4 | Yes |
| `draft_design` | Разработка эскизного проекта | Linear | 3 | Yes |
| `construction_works` | Строительно-монтажные работы | Linear | 4 | Yes |
| `meter_sealing` | Установка/снятие пломбы | Linear | 2 | Yes |

---

## Deployment

Multi-stage `Dockerfile` + `docker-compose.yml`. LightRAG is embedded — the `api` container is the entire stack.

```bash
cp .env.example .env                                  # fill in OPENAI_API_KEY
docker compose up --build -d                          # build + start in background
docker compose logs -f api                            # follow logs
docker compose exec api python ingestion/ingest.py    # ingest documents
docker compose exec api python -m pytest tests/ -v    # run tests inside container
docker compose down                                   # stop (./data + ./logs persist)
```

**Bind-mount permissions (Linux):** the container runs as uid 1000. If `./data` or `./logs` are root-owned, run `sudo chown -R 1000:1000 data logs` once.

> Back up `./data/lightrag/` before any reindex — it contains the full knowledge graph.

---

## Contributing

1. Branch off `main`, name it `feature/` or `fix/`
2. Add tests for any changed behaviour — the suite lives in `tests/`
3. Run `python -m pytest tests/ -v` locally before opening an MR
4. Open a merge/pull request; CI runs lint (ruff) + tests automatically

To add a new service intent: edit `config/services.yaml` + `config/keywords.yaml` only — no Python changes required. See [CLAUDE.md](CLAUDE.md) for flow internals and architectural conventions.

---

## License

[MIT](LICENSE)
