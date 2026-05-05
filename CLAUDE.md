# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Dev commands

```bash
# Activate venv (run once per shell session)
venv/Scripts/activate             # bash/Git Bash
venv\Scripts\Activate.ps1         # PowerShell

# Install dependencies (one-time)
pip install lightrag-hku openai

# Start the API server
uvicorn app.main:app --reload --port 8001

# Ingest documents into LightRAG
python ingestion/ingest.py
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
LLM_PROVIDER=openai         # "openai" only (ollama no longer supported)
OPENAI_MODEL=gpt-4.1-nano   # Model for both entity extraction and LLM responses
```

Old Qdrant/Ollama variables are no longer used.

## Architecture

FastAPI RAG-based AI assistant for the iSEL platform. Responds in Russian (RU) and Kazakh (KZ).

**Demo UI:** `http://localhost:8001/static/index.html` — glassmorphism floating widget (FAB bottom-right) built with Flaticon Uicons and Inter font. No mobile styles.

**Health check:** `GET /health` returns `{"status": "ok", "service": "isel-bot"}`.

### Request pipeline (`POST /api/v1/chat`)

Handled in `app/routers/chat.py`. Input: `ChatRequest(message, session_id, page?, language?)`.  
Response: `ChatResponse(answer, source, handoff)` — `source` is `"rule_based"` or `"llm"`.

The `language` field in `ChatRequest` is accepted but **not used**. Language is auto-detected from the incoming `message` via `_detect_language()`, which checks for Kazakh-specific Cyrillic characters (ә ғ қ ң ө ұ ү і).

Sessions are **in-memory** (`sessions: dict[str, list[dict]] = {}` in `chat.py`) — lost on server restart.

1. **Rule-based filter** (`services/rulebased.py`)  
   Keys in `SYSTEM_COMMANDS` are matched by longest-key-first substring search (prevents short keys shadowing longer ones). Covers greetings, identity, capability, off-topic deflection, and operator handoff in both RU and KZ (`привет`, `салем`, `спасибо`, `рахмет`, etc.). Returns early without hitting RAG or the LLM.

2. **Entity detection and query enrichment** (`routers/chat.py`)  
   `_detect_entity()` scans the current message and all prior user messages for FL/UL keywords in Russian (`физ`, `юр`, `фл`, `юл`, …) and Kazakh (`жеке`, `заңды`, `занды`) terms.  
   - If entity found AND prior history exists: pairs last user question with current message for the RAG query (assumes current message is the clarification).  
   - If entity found but no prior history: appends entity to the RAG query.  
   - LLM always receives `[пользователь уже указал: {entity}]` appended to the question when entity is known.  
   Session history is capped at 10 messages; last 4 are passed to the LLM as `history_text`.

3. **RAG retrieval** (`services/rag.py` → `services/lightrag_service.py`)  
   Uses LightRAG's **knowledge graph-based retrieval** (`hybrid` mode):
   - Query is enriched with intent (e.g., `[tu_application]`) and entity (e.g., `[физическое лицо]`)
   - LightRAG extracts entities and relationships from documents using gpt-4.1-nano
   - Retrieval combines **graph traversal** (follows relationships between concepts like steps, situations, buttons) with **vector similarity** (text-embedding-3-small, 1536-dim)
   - Hybrid mode: local graph neighborhood + global community summaries
   - Returns raw context string with `only_need_context=True` (no LLM generation inside LightRAG)
   - Empty context falls back gracefully; router passes `"Контекст недоступен."` to the LLM if needed

4. **LLM generation** (`services/llm_service.py`)  
   `LLM_PROVIDER=ollama` → `_call_ollama` (POST `/api/generate`, stream=False, timeout 180 s).  
   `LLM_PROVIDER=openai` → `_call_openai` (POST to OpenAI chat completions, timeout 30 s, max_tokens 1000).  
   Temperature is `0.0` in both providers.  
   System prompt uses XML tags (`<role>`, `<language_rule>`, `<history>`, `<rules>`, `<formatting>`).  
   User prompt wraps input as `<context>`, `<question>`, `<answer>` — the open `<answer>` tag primes extraction.  
   When `language="kz"`, `<language_rule>` mandates full Kazakh translation of Russian templates and forbids any Russian in the response.  
   `<formatting>` includes a rule to strip guillemet/quote wrapping from button names and replace with `**bold**`.

### Document API (`routers/documents.py`)

- `POST /api/v1/upload` — saves file with `{uuid}_{original_name}` to `data/docs/`, queues ingestion via `TaskQueue` (max 2 concurrent ingestions)
- `GET /api/v1/documents` — lists files in `data/docs/`
- `DELETE /api/v1/documents/{filename}` — removes file from disk only (does **not** remove KG nodes from LightRAG; LightRAG keeps the graph as-is)

### Ingestion pipeline (`ingestion/ingest.py`)

`ingest()` initializes LightRAG (no collection wipe). Each run **incrementally adds** documents to the knowledge graph.

**Chunking strategy** (`split_by_situations()`):
1. **Meta blocks** (`Цель интента`, `Описание интента`, `Мақсаты`, `Сипаттамасы`, `Требования`, `Талаптар`) — extracted first, added at the end (lower retrieval priority).
2. **Situation blocks** (`Ситуация \d+`, `Жағдай \d+`) — each captures content until the next situation, including `Шаблон ответа`/`Жауап үлгісі` (content preserved, marker stripped). Ensures every block has both description and answer template.
3. **Deduplication** — keeps chunk with more text content when same title appears twice.
4. **Fallback** — fixed-size word chunking (800 words, 100-word overlap) if no situation headers found.

**Each chunk inserted into LightRAG** with metadata:
```
[FILENAME: {filename}] [INTENT: {intent_slug}] [LANGUAGE: {language}] [TITLE: {title}]
{chunk_text}
```

LightRAG automatically:
- Tokenizes by 1200-token chunks with 100-token overlap
- Extracts entities and relations using gpt-4.1-nano
- Embeds chunks with text-embedding-3-small (1536-dim)
- Builds / updates the knowledge graph
- Stores graph in `./data/lightrag/` (NetworkX + nano-vectordb)

## Key constraints

- All packages under `app/` have `__init__.py` — do not delete them.
- `frontend/` directory must exist before starting the server (`StaticFiles` mount fails otherwise).
- `data/docs/` is the staging area for raw documents; deleting a file via the API does **not** clean up its nodes from the LightRAG knowledge graph.
- `data/lightrag/` is the persistent storage for the knowledge graph — back this up; it contains all indexed content.
- `ingest.py` supports `.pdf`, `.txt`, `.md`, `.docx` — all are parsed and sent to LightRAG as situation-level blocks.
- LightRAG ingestion triggers gpt-4.1-nano calls for entity extraction — expect API costs (~0.5–2 USD per 100KB document).
- No test suite yet — add pytest and document commands here when added.
- `OPENAI_API_KEY` **must** be set in `.env` for the app to function (both LLM generation and embeddings use OpenAI).