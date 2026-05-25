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
| `tests/` | pytest test suite (200+ tests, ~95% pass rate) |

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
  ├─ navigation FAQ shortcut (status/download queries → nav path, no LLM)
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
`source` values: `"rule_based"` | `"faq_interruption"` | `"llm"`

---

## Document Upload & Reindex Flow

### Upload (incremental, no auto-index)

```
POST /api/v1/upload
  → file saved as data/docs/{uuid}_{original_name}
  → TaskQueue.submit(_run_ingestion)   ← max 2 concurrent
  → ingest_file() [async]
    → split_by_situations() — situation blocks → meta blocks → 800-word fallback chunks
    → rag.ainsert(chunk) × N   ← adds to KG, never wipes
```

Deleting via `DELETE /api/v1/documents/{filename}` removes the file from disk only — **does not** remove KG nodes from LightRAG.

### Reindex (full wipe + rebuild via SSE)

The `/reindex` endpoint performs a complete rebuild and streams progress via **Server-Sent Events**:

```
POST /api/v1/reindex
  → wipe data/lightrag/ directory entirely
  → reinitialize LightRAG (lightrag_service.initialize())
  → for each file in data/docs/:
      → ingest_file(filepath)         ← try/except per file so one failure doesn't stop all
      → yield SSE: {"status": "progress", "file": name, "done": N, "total": M}
  → yield SSE: {"status": "done"}     ← guaranteed via finally block
```

**SSE event format:**
```
data: {"status": "progress", "file": "...", "done": 1, "total": 5}\n\n
data: {"status": "ping"}\n\n          ← heartbeat every ~15s to keep connection alive
data: {"status": "done"}\n\n          ← always sent, even on error
```

**Frontend** handles `EventSource` `onerror` by calling `source.close()` and unlocking the UI immediately (graceful degradation — never leave the reindex button disabled on failure).

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
| `supply_contract_residential` | `FAQFlow` | — | No | Бытовой договор — pure retrieval |
| `supply_contract_non_residential` | `FAQFlow` | — | No | Небытовой договор — pure retrieval |
| `load_calculation` | `FAQFlow` | — | No | Расчёт нагрузки |
| `draft_design` | `FAQFlow` | — | No | Эскизный проект |
| `construction_works` | `FAQFlow` | — | No | СМР |
| `meter_sealing` | `FAQFlow` | — | No | Установка/снятие пломбы |
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

### SSE — heartbeat/ping

Send `data: {"status": "ping"}\n\n` every ~15 seconds during long operations to prevent proxy/browser timeouts from silently killing the connection.

### Frontend graceful degradation

On `EventSource` `onerror` or explicit `close()`, always unlock the UI (re-enable buttons, hide spinners). Never leave the user with a frozen interface.

```js
source.onerror = () => {
    source.close();
    unlockUI();   // always — regardless of whether done was received
};
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

## External Services

No external vector DB or graph DB. Everything is local:

| Component | Implementation |
|---|---|
| Graph store | NetworkX (embedded, persisted to `data/lightrag/`) |
| Vector DB | nano-vectordb (embedded, persisted to `data/lightrag/`) |
| LLM | OpenAI API (required) |
| Embeddings | OpenAI `text-embedding-3-small` |
