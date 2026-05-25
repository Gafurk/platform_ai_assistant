# AI Assistant

FastAPI RAG chatbot for the **iSEL** platform. Answers user questions in **Russian** and **Kazakh** using a LightRAG knowledge graph (graph + vector hybrid) backed by OpenAI.

- **UI:** `http://localhost:8001/static/index.html`
- **Health:** `GET /health` → `{"status": "ok", "service": "isel-bot"}`

---

## Quick Start

```bash
# 1. Virtual environment
python -m venv venv
venv\Scripts\Activate.ps1        # PowerShell
# source venv/Scripts/activate   # bash / Git Bash

# 2. Dependencies
pip install -r requirements.txt

# 3. Create .env
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-5-mini            # chat responses — reasoning model OK
LIGHTRAG_EXTRACT_MODEL=gpt-4.1-nano  # entity extraction — NON-reasoning only

# 4. Ingest documents into the knowledge graph
python ingestion/ingest.py

# 5. Run
uvicorn app.main:app --reload --port 8001
```

> **`LIGHTRAG_EXTRACT_MODEL` must be a non-reasoning model.** Reasoning models (`o`-series, `gpt-5-mini`) exhaust `max_completion_tokens` on LightRAG's 14 KB extraction prompt and produce 0 entities. `OPENAI_MODEL` (used for chat) can be any model including reasoning ones — `max_completion_tokens=8000` provides enough budget for internal reasoning tokens.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| RAG / KG | LightRAG-HKU (NetworkX graph + nano-vectordb) |
| LLM (chat) | OpenAI `gpt-5-mini` — set via `OPENAI_MODEL`; reasoning models supported |
| LLM (extraction) | OpenAI `gpt-4.1-nano` — set via `LIGHTRAG_EXTRACT_MODEL`; **non-reasoning only** |
| Embeddings | OpenAI `text-embedding-3-small` — hardcoded, dim=1536 |
| Session state | In-memory `SessionManager` — lost on restart |
| Streaming | Server-Sent Events (reindex progress) |
| Config | YAML: `config/services.yaml`, `config/keywords.yaml` |

### Which model does what

| Role | Model | When called |
|---|---|---|
| **Chat responses** | `gpt-5-mini` (`OPENAI_MODEL`) | Every user message that reaches the LLM — `llm_service.py` |
| **Entity/relation extraction** | `gpt-4.1-nano` (`LIGHTRAG_EXTRACT_MODEL`) | Document ingestion only — builds the knowledge graph, `lightrag_service.py` |
| **Embeddings** | `text-embedding-3-small` | Document ingestion (index) + every search query — hardcoded in `lightrag_service.py` |
| **LightRAG retrieval** | _no LLM_ | `only_need_context=True` returns raw chunks from graph+vector, no generation |

---

## Project Structure

```
app/
├── config/
│   └── service_registry.py   # Singleton — loads YAML once at import
├── flows/
│   ├── base.py               # FlowContext dataclass + BaseFlow interface
│   ├── linear.py             # Step-by-step flows (entity-gated where needed)
│   ├── scenario.py           # Branching flows (real estate)
│   ├── faq.py                # Pure retrieval, no state
│   └── registry.py           # get_flow(), classify_intent(), INTENT_KEYWORDS
│                             #   — built dynamically from config/keywords.yaml
├── models/
│   └── schemas.py            # ChatRequest, ChatResponse, FlowState
├── routers/
│   ├── chat.py               # POST /api/v1/chat
│   └── documents.py          # upload / list / delete / reindex
├── services/
│   ├── lightrag_service.py   # LightRAG init + search_docs()
│   ├── llm_service.py        # OpenAI/Ollama call
│   ├── prompt_builder.py     # Static base prompt + dynamic suffix (prefix caching)
│   ├── rulebased.py          # SYSTEM_COMMANDS + NAVIGATION_RULES (zero LLM cost)
│   ├── session.py            # SessionManager
│   ├── rate_limit.py         # Per-session rate limiter
│   ├── task_queue.py         # Async ingestion queue (max 2 concurrent)
│   └── validation.py         # Input validation
└── main.py                   # App entry point, StaticFiles mount

config/
├── services.yaml             # Service definitions, flow type, nav paths
└── keywords.yaml             # Intent keywords (flow_keywords + filter_keywords)

ingestion/
└── ingest.py                 # Document parser + LightRAG inserter

data/
├── docs/                     # Raw document staging (gitignored)
└── lightrag/                 # Persistent KG — back this up before any wipe (gitignored)

frontend/                     # Static UI (glassmorphism widget)
tests/                        # pytest suite, 200+ tests, ~95% pass rate
```

---

## Request Pipeline

```
POST /api/v1/chat
  │
  ├─ validate (ChatRequest)
  ├─ rate_limit (per session_id)
  ├─ rule_based check          ──→ returns immediately if matched
  │    SYSTEM_COMMANDS: greetings, identity, handoff
  │    NAVIGATION_RULES: status/download queries (zero LLM tokens)
  │
  ├─ language detect (Kazakh Cyrillic heuristic: ә ғ қ ң ө ұ ү і)
  ├─ intent classify (sticky → keyword switch → page fallback)
  ├─ navigation FAQ shortcut   ──→ nav path answer, no LLM
  ├─ FAQ interruption gate     ──→ FAQFlow answer, flow state preserved
  ├─ entity detect (ФЛ / ЮЛ from message + history)
  ├─ entity clarification gate (blocks LinearFlow until entity known)
  ├─ flow.next_state()         ──→ LinearFlow / ScenarioFlow / FAQFlow
  ├─ search_docs()             ──→ LightRAG hybrid + intent/step filter
  ├─ ask_llm()                 ──→ OpenAI with cached static prompt + dynamic suffix
  └─ session.update()          ──→ state + history (capped at 10 entries / 5 turns)

Response: ChatResponse(answer, source, handoff)
source values: "rule_based" | "faq_interruption" | "llm"
```

---

## API Reference

### `POST /api/v1/chat`

```json
// Request
{
  "message": "Как подать заявку на ТУ?",
  "session_id": "user-abc",
  "page": "шаг 1",         // optional — used for intent detection on fresh sessions
  "language": "ru"          // optional — overrides auto-detect
}

// Response
{
  "answer": "Чтобы дать точную инструкцию...",
  "source": "llm",
  "handoff": false
}
```

### `GET /health`

```json
{"status": "ok", "service": "isel-bot"}
```

### `POST /api/v1/upload`

Multipart file upload. Saves to `data/docs/`, queues async ingestion (max 2 concurrent).  
Supported formats: `.pdf` `.txt` `.md` `.docx`

### `GET /api/v1/documents`

Lists all files in `data/docs/`.

### `DELETE /api/v1/documents/{filename}`

Removes the file from disk. **Does not** remove KG nodes — those persist until full reindex.

### `POST /api/v1/reindex`

Full wipe of `data/lightrag/` + rebuild from all files in `data/docs/`. Progress streamed via SSE:

```
data: {"status": "progress", "file": "foo.pdf", "done": 1, "total": 5}
data: {"status": "ping"}      ← heartbeat every ~15s
data: {"status": "done"}      ← always sent, even on partial failure
```

---

## Supported Services (14 intents)

All definitions live in `config/services.yaml` + `config/keywords.yaml`. **Adding a new service requires no code changes** — only YAML edits.

| Intent key | Display name (RU) | Flow | Steps | Entity gate |
|---|---|---|---|---|
| `tu_application` | Заявление на технические условия | Linear | 5 | Yes |
| `primary_connection_residential` | Первичное подключение (бытовой) | Linear | 4 | No |
| `primary_connection_nonresidential` | Первичное подключение (небытовой) | Linear | 5 | No |
| `secondary_connection` | Вторичное подключение | Linear | 3 | Yes |
| `contract_termination` | Расторжение договора | Linear | 3 | Yes |
| `grid_disconnection` | Отключение от электросетей | Linear | 2 | Yes |
| `equipment_testing` | Испытание, измерение электрооборудования | Linear | 4 | Yes |
| `real_estate` | Добавление объекта недвижимости | Scenario | — | No |
| `supply_contract_residential` | Договор электроснабжения (бытовой) | FAQ | — | No |
| `supply_contract_non_residential` | Договор электроснабжения (небытовой) | FAQ | — | No |
| `load_calculation` | Расчёт электрической нагрузки | FAQ | — | No |
| `draft_design` | Разработка эскизного проекта | FAQ | — | No |
| `construction_works` | Строительно-монтажные работы | FAQ | — | No |
| `meter_sealing` | Установка/снятие пломбы | FAQ | — | No |

**Entity gate** — LinearFlow blocks at step 0 until user identifies as ФЛ (individual) or ЮЛ (legal entity). Step 1 form fields differ by entity type.

---

## Adding a New Service

No code changes required:

1. Add a block to `config/services.yaml` — set `flow.type` to `linear`, `scenario`, or `faq`.
2. Add matching `flow_keywords` and `filter_keywords` to `config/keywords.yaml`.
3. Upload documents structured with `Ситуация N` / `Шаг N` headers via `POST /api/v1/upload` or `python ingestion/ingest.py`.

`ServiceRegistry` is a singleton; `registry.py` builds `_FLOWS` and `INTENT_KEYWORDS` dynamically from config on startup.

> To add a **new flow type** (new Python class): inherit `BaseFlow`, override `next_state()` + `build_query()`, register in `registry.py`. See `CLAUDE.md`.

---

## Prompt Architecture (OpenAI Prefix Caching)

```
┌─ STATIC BASE — identical per language → cache hit on every request ──────┐
│  <role>  <language_rule>  <intent_handling>                              │
│  <navigation_facts>       <rules>  <formatting>                          │
└──────────────────────────────────────────────────────────────────────────┘
┌─ DYNAMIC SUFFIX — built per request ─────────────────────────────────────┐
│  <history>        last 3 turns                                           │
│  <intent_context> intent / entity / page / situation                    │
│  <step_control>   current step number + "Далее" button instruction      │
└──────────────────────────────────────────────────────────────────────────┘
```

Navigation facts are **auto-generated** from `ServiceRegistry`. Add new paths to `services.yaml → navigation_path`, not to `prompt_builder.py`.

---

## Document Chunking

`split_by_situations()` in `ingestion/ingest.py`:

1. **Situation blocks** — `Ситуация \d+` / `Жағдай \d+` / `Шаг \d+` → one LightRAG insert each.
2. **Meta blocks** — `Цель интента` / `Мақсаты` / `Требования` → lower priority.
3. **Deduplication** — keeps highest-quality version of duplicate situations.
4. **Fallback** — 800-word fixed chunks (overlap=100) if no structured headers found.

Each chunk gets a metadata header:
```
[FILENAME: file.pdf] [INTENT: tu_application] [LANGUAGE: ru] [TITLE: Ситуация 1]
```

> Ingestion cost: ~0.5–2 USD per 100 KB document (entity extraction via OpenAI).

---

## LightRAG Search Pipeline

`search_docs()` in `app/services/lightrag_service.py`:

1. **Query enrichment** — prepend `[TITLE: Шаг N]`, `[INTENT: slug]`, `[entity]`.
2. **Hybrid search** — graph traversal + vector similarity (`top_k=20`, `only_need_context=True`).
3. **Intent filter** — keep only chunks whose `[INTENT:]` tag matches current intent.
4. **Step filter** — if `current_step` is set, keep only chunks containing `Шаг N`; return `""` if none (triggers fallback).
5. **Step fallback** — retry with `"Шаг N {query}"`, `top_k*2`, no metadata prefix.

Two separate OpenAI clients: `_oai_embeddings` (timeout 300s) and `_oai_llm` (timeout 600s). Kept separate because a shared short-timeout client would abort slow entity extractions mid-run.

---

## Running Tests

```bash
python -m pytest tests/ -v

# Single module
python -m pytest tests/test_service_registry.py -v
```

---

## Environment Variables

| Variable | Required | Default | Notes |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | |
| `LLM_PROVIDER` | No | `openai` | `openai` or `ollama` |
| `OPENAI_MODEL` | No | `gpt-4.1-nano` | Chat responses (`llm_service.py`) — reasoning models supported (e.g. `gpt-5-mini`) |
| `LIGHTRAG_EXTRACT_MODEL` | No | `gpt-4.1-nano` | Entity/relation extraction during ingestion — **non-reasoning only** |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama fallback |
| `OLLAMA_MODEL` | No | `llama3.2` | Ollama fallback |
