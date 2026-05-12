# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Dev commands

```bash
# Activate venv (run once per shell session)
venv/Scripts/activate             # bash/Git Bash
venv\Scripts\Activate.ps1         # PowerShell

# Install dependencies (one-time)
pip install lightrag-hku openai pypdf python-docx

# Start the API server
uvicorn app.main:app --reload --port 8001

# Ingest documents into LightRAG
python ingestion/ingest.py

# Run tests
python -m pytest tests/ -v
```

## External services

**NO external services required.** LightRAG uses:
- **Local graph store** — embedded NetworkX graph + persistent storage in `./data/lightrag/`
- **Local vector DB** — embedded nano-vectordb in `./data/lightrag/`
- **OpenAI API** — gpt-4.1-nano for LLM, text-embedding-3-small for embeddings

The working directory `./data/lightrag/` is created automatically on first run.

## .env variables

```
OPENAI_API_KEY=sk-...       # Required — your OpenAI API key
LLM_PROVIDER=openai         # "openai" only (ollama legacy, not recommended)
OPENAI_MODEL=gpt-4.1-nano   # Model for both entity extraction and LLM responses
```

## Architecture

FastAPI RAG-based AI assistant for the iSEL platform. Responds in Russian (RU) and Kazakh (KZ).

**Demo UI:** `http://localhost:8001/static/index.html` — glassmorphism floating widget (FAB bottom-right).

**Health check:** `GET /health` returns `{"status": "ok", "service": "isel-bot"}`.

### Request pipeline (`POST /api/v1/chat`)

```
validate → rate_limit → rule_based → FAQ interruption? → intent classify
→ entity detect → clarification gate → flow.next_state() → RAG → LLM
```

Input: `ChatRequest(message, session_id, page?, language?)`.
Response: `ChatResponse(answer, source, handoff)` — `source` is `"rule_based"`, `"faq_interruption"`, or `"llm"`.

Language is **auto-detected** from the incoming `message` via `_detect_language()`, which checks for Kazakh-specific Cyrillic characters (ә ғ қ ң ө ұ ү і). The `language` field in `ChatRequest` is accepted but not used.

Sessions are **in-memory** (lost on server restart). Managed by `app/services/session.py` `SessionManager`.

### Layer 1 — Rule-based filter (`services/rulebased.py`)

Two dictionaries checked before any RAG or LLM call:

**`SYSTEM_COMMANDS`** — identity, greetings, handoff  
Keys matched by longest-first substring search. Covers `привет`, `салем`, `спасибо`, `рахмет`, `оператор`, etc.

**`NAVIGATION_RULES`** — canonical platform navigation paths  
Exact answers for status/document/refusal queries — zero LLM tokens.

| Trigger phrase | Answer |
|---|---|
| `статус заявки`, `статус обращения`, `где моя заявка` | Path to Услуги → Обращения |
| `мотивированный отказ`, `скачать отказ` | Path + Скачать/Просмотр instruction |
| `скачать технические условия`, `скачать тУ`, `готовые тУ` | Path + Скачать/Просмотр instruction |
| Kazakh equivalents | Same answers in Kazakh |

**Rule:** Do NOT add navigation paths to `app/flows/`. Flows are state machines. Navigation facts belong in `rulebased.py` (for exact matches) and `llm_service.py` `<navigation_facts>` (for contextual use).

### Layer 2 — Flow engine (`app/flows/`)

**`FlowState`** (in `models/schemas.py`): `intent`, `entity`, `step`, `situation`, `locked`

**Flow types:**

| Intent | Flow class | Behavior |
|---|---|---|
| `tu_application` | `LinearFlow` | 5 steps, entity required, step increments on "дальше"/"да"/etc. |
| `real_estate` | `ScenarioFlow` | Branching situations, asks clarifying question, uses last-turn context in query |
| `None` | `FAQFlow` | Pure retrieval, no state tracking |

**FAQ interruption:** When `state.locked=True` and the message has no flow keywords and is not a step phrase, the system answers via `FAQFlow` without disturbing the active flow state. History-only update preserves the step position.

**Intent locking:** Once step≥1 starts for TU or situation is active for real estate, `state.locked=True` prevents accidental intent resets.

**`registry.py`** exports:
- `get_flow(intent)` — returns the flow handler
- `classify_intent(message, page, current_intent)` — sticky intent with explicit-switch override
- `has_flow_keywords(message)` — used by FAQ interruption gate

### Layer 3 — RAG retrieval (`services/rag.py` → `services/lightrag_service.py`)

LightRAG `hybrid` mode — combines graph traversal + vector similarity.

Query enrichment order: `[Шаг N]` → `[intent]` → `[entity]` → user message.

`rerank_model_func=None` — reranking disabled (no reranker model available).

Returns raw context string (`only_need_context=True`). Empty context falls back to `"Контекст недоступен."`.

### Layer 4 — LLM generation (`services/llm_service.py`)

**Prompt structure** (for OpenAI prefix caching):

```
_STATIC_PROMPT_RU / _STATIC_PROMPT_KZ   ← identical per language — cached by OpenAI
  <role>
  <language_rule>
  <navigation_facts>                      ← canonical paths for status/documents
  <intent_handling>
  <rules> (13 rules)
  <formatting>

  + per-request dynamic suffix:
  <history>         (if any)
  <intent_context>  (if intent/entity/page set)
  <step_control>    (if current_step set)
    → step 5: includes navigation hint to Услуги → Обращения
```

Temperature `0.0`. Max tokens `1000`. Timeout `30s` (OpenAI) / `180s` (Ollama).

### Session management (`services/session.py`)

`SessionManager` has three update modes:

| Method | When | Effect |
|---|---|---|
| `update()` | Normal LLM turn | State + history |
| `update_state_only()` | Clarification gate (no bot message) | State only |
| `update_history_only()` | FAQ interruption | History only (flow state preserved) |

History capped at 20 entries (10 turns). `cleanup()` removes sessions idle > 24 hours.

### Document API (`routers/documents.py`)

- `POST /api/v1/upload` — saves file with `{uuid}_{original_name}` to `data/docs/`, queues ingestion via `TaskQueue` (max 2 concurrent)
- `GET /api/v1/documents` — lists files in `data/docs/`
- `DELETE /api/v1/documents/{filename}` — removes file from disk only (**does not** remove KG nodes from LightRAG)

### Ingestion pipeline (`ingestion/ingest.py`)

Each run **incrementally adds** documents to the knowledge graph (no wipe).

**Chunking strategy** (`split_by_situations()`):
1. **Situation blocks** (`Ситуация \d+`, `Жағдай \d+`) — each gets its own LightRAG insert with metadata header.
2. **Meta blocks** (`Цель интента`, `Мақсаты`, `Требования`) — added at lower priority.
3. **Fallback** — 800-word fixed chunks with 100-word overlap if no situation headers found.

Each chunk inserted with:
```
[FILENAME: {filename}] [INTENT: {intent_slug}] [LANGUAGE: {language}] [TITLE: {title}]
{chunk_text}
```

## Key constraints

- All packages under `app/` have `__init__.py` — do not delete them.
- `frontend/` directory must exist before starting the server (`StaticFiles` mount fails otherwise).
- `data/docs/` is the staging area for raw documents; deleting via API does **not** clean LightRAG KG nodes.
- `data/lightrag/` is the persistent KG storage — **back this up**; it contains all indexed content.
- LightRAG ingestion triggers gpt-4.1-nano calls — expect ~0.5–2 USD per 100KB document.
- `OPENAI_API_KEY` **must** be set in `.env`.

## Adding new navigation rules

When a new platform path needs to be surfaced to users:

1. Add the answer string to `rulebased.py` `NAVIGATION_RULES` dict.
2. Add both Russian and Kazakh trigger phrases.
3. If the path is relevant during LLM flows (not just direct questions), also add to `<navigation_facts>` in `llm_service.py` static prompts.
4. Do **not** add navigation content to `app/flows/` — flows handle routing only.

## Adding a new flow type

1. Create `app/flows/myflow.py` inheriting `BaseFlow`.
2. Override `next_state()`, `build_query()`, and optionally `is_step_progression()`.
3. Register in `app/flows/registry.py` `_FLOWS` dict.
4. Add intent keywords to `INTENT_KEYWORDS` in `registry.py`.
5. Add tests in `tests/test_flows.py`.
