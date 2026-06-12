# CLAUDE.md

Guidance for Claude Code sessions working in this repository.

---

## Project Overview

FastAPI RAG-based AI assistant for the **iSEL** platform. Responds in Russian and Kazakh. Uses LightRAG (graph + vector hybrid) backed by OpenAI for entity extraction and embeddings. Entry point: `app/main.py`.

Demo UI: `http://localhost:8001/static/index.html` — glassmorphism floating chat widget.  
Health check: `GET /health` → `{"status": "ok", "service": "isel-bot"}`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| RAG / KG | LightRAG-HKU (hybrid: graph traversal + nano-vectordb) |
| LLM / Embeddings | OpenAI (`gpt-4.1-nano` for extraction, configurable model for chat) |
| Session state | In-memory (`SessionManager`) — lost on restart |
| Frontend | Vanilla HTML/CSS/JS (glassmorphism) |
| Streaming | Server-Sent Events (SSE) for reindex status |
| Config | YAML files (`config/services.yaml`, `config/keywords.yaml`) |

---

## Dev Commands

```bash
# Activate venv (run once per shell session)
venv\Scripts\Activate.ps1           # PowerShell
venv/Scripts/activate               # bash/Git Bash

# Install dependencies (one-time)
pip install -r requirements.txt

# Start API server
uvicorn app.main:app --reload --port 8001

# Run tests
python -m pytest tests/ -v

# Manual bulk ingestion (from scratch or after reindex wipe)
python ingestion/ingest.py
```

### Docker (recommended for running / deploying)

```bash
cp .env.example .env                                  # set OPENAI_API_KEY
docker compose up --build                             # build + start (add -d for background)
docker compose logs -f api                            # follow logs
docker compose exec api python ingestion/ingest.py    # bulk-ingest data/docs/ into the KG
docker compose exec api python -m pytest tests/ -v    # run the test suite inside the container
docker compose down                                   # stop (./data + ./logs persist on host)
```

`api` is the only container — LightRAG is embedded (NetworkX + nano-vectordb), so there is no external DB. `./data` and `./logs` are bind-mounted for persistence. See **Deployment (Docker)** below.

---

## Environment Variables (`.env`)

```env
OPENAI_API_KEY=sk-...                    # Required
LLM_PROVIDER=openai                      # "openai" (default) or "ollama"
OPENAI_MODEL=gpt-5-mini                  # Model for LLM chat responses — can be reasoning
LIGHTRAG_EXTRACT_MODEL=gpt-4.1-nano      # Model for LightRAG entity/relation extraction
                                         # Must be a NON-reasoning model — reasoning models
                                         # exhaust max_completion_tokens on LightRAG's 14 KB
                                         # system prompt before producing any output.
OLLAMA_BASE_URL=http://localhost:11434   # Fallback only
OLLAMA_MODEL=llama3.2                    # Fallback only
```

Copy `.env.example` → `.env` and fill in `OPENAI_API_KEY`. Locally, `python-dotenv` loads `.env` at startup; under Docker Compose it is injected via `env_file`. `.env` is gitignored and excluded from the image by `.dockerignore` — never bake secrets into the build.

---

## Important Directories

| Path | Purpose |
|---|---|
| `app/routers/` | FastAPI route handlers (`chat.py`, `documents.py`) |
| `app/services/` | Business logic: LightRAG, LLM, session, rule-based, rate limit |
| `app/flows/` | Intent-based state machines (`linear.py`, `scenario.py`, `faq.py`) |
| `app/config/` | `ServiceRegistry` singleton — loads YAML configs once at import |
| `config/` | `services.yaml` (services + nav paths), `keywords.yaml` (intent keywords) |
| `ingestion/` | Document chunking + LightRAG insertion pipeline |
| `frontend/` | Static UI — **must exist** before server start (StaticFiles mount fails otherwise) |
| `data/docs/` | Staging area for uploaded raw documents |
| `data/lightrag/` | **Persistent KG storage** (graph + embeddings) — **back this up** |
| `tests/` | pytest test suite (300+ tests) |
| `Dockerfile` | Multi-stage build (`python:3.11-slim`, non-root `appuser`) |
| `docker-compose.yml` | Single `api` service; mounts `./data` + `./logs`; loads `.env` |
| `.dockerignore` | Excludes `data/`, `logs/`, `venv/`, `tests/`, `.git/`, `.env` from build context |
| `.env.example` | Template — copy to `.env` and set `OPENAI_API_KEY` |

---

## Architecture & Request Pipeline

```
POST /api/v1/chat
  │
  ├─ validate (ChatRequest)
  ├─ rate_limit (per session_id)
  ├─ rule_based → SYSTEM_COMMANDS / NAVIGATION_RULES   ← returns immediately if matched
  ├─ language detect (_detect_language — Kazakh Cyrillic: ә ғ қ ң ө ұ ү і)
  ├─ intent classify (sticky + keyword switch + page fallback)
  ├─ navigation FAQ shortcut (status/download queries → LightRAG + ask_llm, source "llm"; state preserved)
  ├─ FAQ interruption gate (locked flow + no flow keywords → FAQFlow, preserve state)
  ├─ entity detect (ФЛ/ЮЛ from message + history)
  ├─ entity clarification gate (TU application blocks until entity known)
  ├─ flow.next_state() (LinearFlow / ScenarioFlow / FAQFlow)
  ├─ flow lock (state.locked=True once step≥1 or situation chosen)
  ├─ search_docs() — LightRAG hybrid + intent/step filter + step fallback
  ├─ ask_llm() — OpenAI with static cached prompt + dynamic suffix
  └─ session.update() (state + history)
```

**Response:** `ChatResponse(answer, source, handoff)`  
`source` values actually returned: `"validation"` | `"rate_limit"` | `"rule_based"` | `"llm"`. Note `"faq_interruption"` is only a **log label** (`log_chat_request`), not a response `source` — both the navigation shortcut and the FAQ interruption return `source="llm"`.  
`handoff` is **always `false`** today: `check_rule()` computes an operator/handoff flag, but the chat router discards it (`rule_answer, _ = check_rule(...)`) and hard-codes `handoff=False` on every return path. Wire it through if handoff is needed.

---

## Document Upload & Reindex Flow

### Upload (save only — NO auto-index)

```
POST /api/v1/upload
  → validate extension (.pdf/.txt/.md/.docx)
  → file saved as data/docs/{uuid4().hex}_{original_name}
  → return {"status": "saved", "filename": ...}
```

**Upload does not index.** It only writes the file to disk — there is no background ingestion. The knowledge graph is (re)built solely by `POST /api/v1/reindex` or `python ingestion/ingest.py`.

> `app/services/task_queue.py` defines a `TaskQueue` (and an `ingestion_queue = TaskQueue(max_concurrent=2)` singleton), but nothing calls `.submit()` — it is currently **dead/unused code**. Do not document upload as queue-driven.

Deleting via `DELETE /api/v1/documents/{filename}` removes the file from disk only — **does not** remove KG nodes from LightRAG.

### Reindex (full wipe + rebuild via SSE)

The `/reindex` endpoint performs a complete rebuild and streams progress via **Server-Sent Events**:

```
POST /api/v1/reindex
  → yield {"status": "wiping"} then {"status": "init"}
  → lightrag_service.reinitialize()   ← rmtree data/lightrag/ + fresh LightRAG instance
  → files = sorted(data/docs/*)
  → for i, file in enumerate(files):
      → yield {"status": "progress", "file": name, "done": i, "total": M}   ← done is 0-based
      → ingest_file(filepath, name)   ← try/except per file; on failure yield "file_error" and continue
  → yield {"status": "done", "total": M}
  (top-level exception → yield {"status": "error", "message": ...} then {"status": "done"})
```

**Actual SSE statuses** (no others exist):
```
data: {"status": "wiping"}\n\n
data: {"status": "init"}\n\n
data: {"status": "progress", "file": "...", "done": 0, "total": 5}\n\n
data: {"status": "file_error", "file": "...", "error": "...", "done": 2, "total": 5}\n\n
data: {"status": "done", "total": 5}\n\n          ← reached on success and after a top-level error
data: {"status": "error", "message": "..."}\n\n   ← only on top-level failure
```

> **There is no `ping`/heartbeat event.** Earlier docs claimed one; it is not implemented. Don't rely on it.

**Frontend** consumes the stream with `fetch()` + `res.body.getReader()` + `TextDecoder` (**not** `EventSource`). The reindex button is always re-enabled in a `finally` block, so a dropped connection never leaves the UI locked.

---

## Chunking Strategy (`split_by_situations()`)

1. **Situation blocks** (`Ситуация \d+`, `Жағдай \d+`, `Шаг \d+`) — each block is one LightRAG insert.
2. **Meta blocks** (`Цель интента`, `Мақсаты`, `Требования`) — lower priority.
3. **Deduplication** — keeps highest-quality version of duplicate situations.
4. **Fallback** — 800-word fixed chunks (overlap=100) if no headers found.

Each chunk is inserted with a metadata header:
```
[FILENAME: file.pdf] [INTENT: tu_application] [LANGUAGE: ru] [TITLE: Ситуация 1]
{chunk text}
```

---

## LightRAG Search Pipeline

`search_docs()` in `app/services/lightrag_service.py`:

1. **Query enrichment** — prepend `[TITLE: Шаг N]`, `[INTENT: slug]`, `[entity]` to query.
2. **Hybrid search** — graph traversal + vector similarity (`only_need_context=True`, `top_k=20`).
3. **Intent filter** (`_filter_context_by_intent`) — keep only chunks whose `[INTENT:]` tag matches.
4. **Step filter** — if `current_step` set, further keep only chunks containing `Шаг N`; return `""` if none (triggers fallback).
5. **Step fallback** — if `filtered_chunk_count == 0`: retry with plain `"Шаг N {query}"`, `top_k*2`, no metadata prefix.

`rerank_model_func=None` — reranking disabled (no reranker available).

**Why two OpenAI clients in `lightrag_service.py`?**  
`_oai_embeddings` (timeout=300s) and `_oai_llm` (timeout=600s) are separate because LightRAG's entity extraction can be slow on large documents; a shared short-timeout client would abort extractions mid-run.

---

## Model Usage

Three distinct model roles — each configured independently:

| Role | Model | Where | Notes |
|---|---|---|---|
| **Chat responses** | `gpt-5-mini` (`OPENAI_MODEL`) | `llm_service.py` | Can be any model incl. reasoning |
| **Entity/relation extraction** | `gpt-4.1-nano` (`LIGHTRAG_EXTRACT_MODEL`) | `lightrag_service.py → gpt41_nano_complete()` | **Must be non-reasoning** — called only at document ingestion time |
| **Embeddings** | `text-embedding-3-small` | `lightrag_service.py → openai_embed()` | Hardcoded; dim=1536; used at ingestion and at query time |

**LightRAG search itself calls no LLM.** `only_need_context=True` makes LightRAG return raw chunks (graph + vector retrieval) without generating an answer. The LLM is only invoked separately via `ask_llm()` in the chat router.

## LLM Service (`app/services/llm_service.py`)

- **OpenAI:** `temperature=1`, `max_completion_tokens=8000`, timeout=90s.
- **Ollama:** `temperature=0.0`, timeout=180s.

8000 tokens required for reasoning models (gpt-5-mini, o-series): they consume 3000–5000 internal reasoning tokens before producing visible output.

---

## Prompt Architecture (OpenAI prefix caching)

```
STATIC BASE (identical per language → OpenAI cache hit):
  <role> <language_rule> <intent_handling> <navigation_facts> <rules> <formatting>

DYNAMIC SUFFIX (per request):
  <history>         last 3 turns
  <intent_context>  intent / entity / page / situation
  <step_control>    current step number + "Далее" button instruction
```

Navigation facts are **auto-generated** from `ServiceRegistry` (reads `services.yaml`). Add new paths to `services.yaml`, not to `prompt_builder.py` directly.

---

## Flow System

### Flow types

| Intent | Class | Steps | Entity gate | Behavior |
|---|---|---|---|---|
| `tu_application` | `LinearFlow` | 5 | Yes | Entity (ФЛ/ЮЛ) required before step 1; advances on "дальше"/"да" |
| `primary_connection_residential` | `LinearFlow` | 4 | No | Бытовое первичное подключение |
| `primary_connection_nonresidential` | `LinearFlow` | 5 | No | Небытовое первичное подключение |
| `secondary_connection` | `LinearFlow` | 3 | Yes | Вторичное подключение; ФЛ/ЮЛ form differs at step 1 |
| `contract_termination` | `LinearFlow` | 3 | Yes | Расторжение договора; signing via ЭЦП or SMS |
| `grid_disconnection` | `LinearFlow` | 2 | Yes | Отключение от электросетей |
| `equipment_testing` | `LinearFlow` | 4 | Yes | Испытание/измерение электрооборудования; step 4 = поставщик selection |
| `real_estate` | `ScenarioFlow` | — | No | Branching (cadastre vs address register); stores `original_question` |
| `supply_contract_residential` | `LinearFlow` | — | No | Бытовой договор — pure retrieval |
| `supply_contract_non_residential` | `LinearFlow` | — | No | Небытовой договор — pure retrieval |
| `load_calculation` | `LinearFlow` | — | No | Расчёт нагрузки |
| `draft_design` | `LinearFlow` | — | No | Эскизный проект |
| `construction_works` | `LinearFlow` | — | No | СМР |
| `meter_sealing` | `LinearFlow` | — | No | Установка/снятие пломбы |
| `None` | `FAQFlow` | — | No | General FAQ; no state tracking |

### Session update modes

| Method | When | Effect |
|---|---|---|
| `update()` | Normal LLM turn | State + history |
| `update_state_only()` | Clarification gate (no bot answer yet) | State only |
| `update_history_only()` | FAQ interruption (flow active) | History only — flow state preserved |

History capped at 10 entries (5 turns).

---

## Layer 1 — Rule-Based Filter (`services/rulebased.py`)

Two dicts checked **before** any RAG or LLM call:

- **`SYSTEM_COMMANDS`** — identity, greetings, handoff. Matched by longest-first substring.
- **`NAVIGATION_RULES`** — exact answers for status/document/refusal queries (zero LLM tokens).

**Rule:** Do NOT put navigation paths in `app/flows/`. Flows handle routing only. Navigation facts belong in:
1. `rulebased.py` `NAVIGATION_RULES` (exact match, no LLM).
2. `services.yaml` `navigation_path` (used by prompt builder for contextual LLM flows).

---

## Adding New Navigation Rules

1. Add answer string to `NAVIGATION_RULES` in `rulebased.py`.
2. Add Russian and Kazakh trigger phrases.
3. If relevant during LLM flows, also add to `services.yaml` → `navigation_path`.
4. Do **not** add to `app/flows/`.

## Adding a New Flow Type

1. Create `app/flows/myflow.py` inheriting `BaseFlow`.
2. Override `next_state()`, `build_query()`, optionally `is_step_progression()`.
3. Register in `app/flows/registry.py` `_FLOWS` dict.
4. Add intent keywords to `INTENT_KEYWORDS` in `registry.py`.
5. Add tests in `tests/test_flows.py`.

---

## Code Style & Conventions

These rules are **strict** — apply them to all new and modified code.

### Async correctness

Always use `await asyncio.sleep()`. Never `time.sleep()` inside async functions — it blocks the event loop and freezes all SSE connections.

### Batch loop error isolation

Wrap each iteration of a batch/file loop in its own `try/except`. One bad document must not abort an entire reindex or ingestion run.

```python
for file in files:
    try:
        await ingest_file(file)
    except Exception as e:
        logger.error(f"Failed {file}: {e}")
        continue
```

### SSE — always send `done`

The `done` event **must** be sent on every SSE code path: success, partial failure, and uncaught exception. Use `finally` as a safety net. If the frontend never receives `done`, the UI stays locked indefinitely.

```python
async def reindex_stream():
    try:
        # ... work ...
        yield 'data: {"status": "done"}\n\n'
    except Exception as e:
        yield f'data: {{"status": "error", "message": "{e}"}}\n\n'
        yield 'data: {"status": "done"}\n\n'
```

### SSE — heartbeat/ping (recommended, NOT currently implemented)

A `data: {"status": "ping"}\n\n` heartbeat every ~15s during long operations helps prevent proxy/browser timeouts from silently killing the connection. **The current `/reindex` stream does not send one** — add it here if you observe dropped connections on slow reindexes. (The `wiping`/`init`/`progress` events already provide some traffic in practice.)

### Frontend graceful degradation

The reindex client uses `fetch()` + `res.body.getReader()` (a `ReadableStream`), **not** `EventSource`. Always re-enable the UI (button, spinner) in a `finally` block so a dropped or errored stream never leaves a frozen interface.

```js
try {
  const reader = res.body.getReader();
  // ... read/parse "data: {...}" lines ...
} catch (e) {
  // show error
} finally {
  btn.disabled = false;   // always — regardless of whether "done" was received
}
```

---

## Key Constraints

- All packages under `app/` have `__init__.py` — do not delete them.
- `frontend/` directory **must** exist before server start (`StaticFiles` mount raises on missing dir).
- `data/lightrag/` is the persistent KG — **back it up before any wipe**.
- LightRAG ingestion triggers OpenAI calls (entity extraction) — ~0.5–2 USD per 100 KB document.
- `DELETE /api/v1/documents/{filename}` removes the file only — KG nodes remain until full reindex.
- `LIGHTRAG_EXTRACT_MODEL` **must** be a non-reasoning model. Reasoning models exhaust `max_completion_tokens` on LightRAG's 14 KB extraction prompt before writing any output → 0 entities extracted.
- Do not add navigation paths to `app/flows/` — flows handle state transitions only.

---

## Deployment (Docker)

Multi-stage `Dockerfile` + `docker-compose.yml`. Because LightRAG is embedded, the `api` container is the entire stack — no external vector/graph DB.

- **Build** — deps are installed into an isolated venv in a builder stage, then copied into a slim `python:3.11-slim` runtime image (smaller final image, no build toolchain shipped).
- **Runtime** — runs as unprivileged `appuser` (uid 1000); `uvicorn app.main:app` on `0.0.0.0:8001`, **no `--reload`**.
- **Persistence** — `./data` (KG + uploaded docs) and `./logs` are bind-mounted. **Back up `./data/lightrag` before any reindex/wipe.**
- **Config** — `.env` loaded via Compose `env_file`.
- **Healthcheck** — container polls `GET /health` every 30s.

> **Bind-mount permissions (Linux).** The container writes as uid 1000. If `./data` or `./logs` end up root-owned and you hit "permission denied", run `sudo chown -R 1000:1000 data logs` once (or create the dirs before first start). Docker Desktop on macOS/Windows handles this automatically.

> **Editing `requirements.txt`** invalidates the dependency layer — rebuild with `docker compose up --build`.

---

## External Services

No external vector DB or graph DB. Everything is local:

| Component | Implementation |
|---|---|
| Graph store | NetworkX (embedded, persisted to `data/lightrag/`) |
| Vector DB | nano-vectordb (embedded, persisted to `data/lightrag/`) |
| LLM | OpenAI API (required) |
| Embeddings | OpenAI `text-embedding-3-small` |
