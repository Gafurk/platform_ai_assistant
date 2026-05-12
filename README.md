# iSEL AI Assistant

FastAPI-based RAG chatbot for the iSEL platform. Answers user questions in **Russian** and **Kazakh** using a LightRAG knowledge graph built from platform documentation.

## Quick start

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\Activate.ps1        # PowerShell
# source venv/Scripts/activate   # bash / Git Bash

# 2. Install dependencies
pip install lightrag-hku openai pypdf python-docx fastapi uvicorn httpx python-dotenv

# 3. Create .env
echo OPENAI_API_KEY=sk-...  > .env
echo LLM_PROVIDER=openai   >> .env
echo OPENAI_MODEL=gpt-4.1-nano >> .env

# 4. Ingest documents
python ingestion/ingest.py

# 5. Start server
uvicorn app.main:app --reload --port 8001
```

Chat UI: [http://localhost:8001/static/index.html](http://localhost:8001/static/index.html)

## API

### `POST /api/v1/chat`

```json
{
  "message": "Как подать заявку на ТУ?",
  "session_id": "user-abc",
  "page": "шаг 1",
  "language": "ru"
}
```

Response:
```json
{
  "answer": "Чтобы дать точную инструкцию...",
  "source": "llm",
  "handoff": false
}
```

`source` values: `"rule_based"` | `"faq_interruption"` | `"llm"` | `"rate_limit"` | `"validation"`

### `GET /health`

```json
{"status": "ok", "service": "isel-bot"}
```

### `POST /api/v1/upload`

Upload a `.pdf`, `.txt`, `.md`, or `.docx` document to be ingested into the knowledge graph.

### `GET /api/v1/documents`

List all documents in `data/docs/`.

### `DELETE /api/v1/documents/{filename}`

Remove a document file. Does **not** remove its nodes from the knowledge graph.

## Project structure

```
app/
  flows/
    base.py         ← FlowContext dataclass + BaseFlow interface
    linear.py       ← TU application (5 steps, entity-gated)
    scenario.py     ← Real Estate (branching situations)
    faq.py          ← General FAQ (no state tracking)
    registry.py     ← get_flow(), classify_intent(), has_flow_keywords()
  models/
    schemas.py      ← ChatRequest, ChatResponse, FlowState
  routers/
    chat.py         ← Thin request handler
    documents.py    ← Document upload/list/delete
  services/
    lightrag_service.py  ← LightRAG init + query wrapper
    llm_service.py       ← System prompt builder + OpenAI/Ollama call
    rag.py               ← search_docs() passthrough
    rulebased.py         ← SYSTEM_COMMANDS + NAVIGATION_RULES
    session.py           ← SessionManager (state + history)
    validation.py        ← Input validation
    rate_limit.py        ← Per-session rate limiter
    task_queue.py        ← Async ingestion queue
  utils/
    logger.py

ingestion/
  ingest.py         ← Document parser + LightRAG inserter

data/
  docs/             ← Staging area for raw documents (gitignored)
  lightrag/         ← Persistent KG storage (gitignored)

frontend/           ← Static UI (glassmorphism widget)
tests/
  test_flows.py     ← 36 flow + registry tests
  test_session.py   ← 13 session management tests
```

## How the bot answers

```
User message
    │
    ├─ Rule-based? (greetings, navigation, handoff) ──→ instant answer
    │
    ├─ FAQ interruption?
    │   (locked flow + no flow keywords + not "дальше") ──→ FAQ answer, flow preserved
    │
    ├─ Intent classify → entity detect → clarification gate
    │
    ├─ Flow.next_state() (step advance / situation tracking)
    │
    ├─ LightRAG hybrid retrieval (graph + vector)
    │
    └─ LLM generation (gpt-4.1-nano, T=0.0)
```

## Navigation rules (zero LLM cost)

The following queries are answered directly from `rulebased.py` without hitting RAG or the LLM:

| User asks about | Answer |
|---|---|
| Application status | **Личный кабинет → Услуги → вкладка Обращения** |
| Motivated refusal / rejection | Same path + **Скачать** / **Просмотр** in the table row |
| Downloading ready TU | Same path + **Скачать** / **Просмотр** when status is «Выполнено» |

## Supported document formats

`.pdf` · `.txt` · `.md` · `.docx`

Documents must be structured with `Ситуация N` or `Жағдай N` headers to be split into granular blocks. Without these headers, fixed-size chunking is used as fallback.

## Running tests

```bash
python -m pytest tests/ -v
```

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `LLM_PROVIDER` | No | `openai` | `openai` or `ollama` (legacy) |
| `OPENAI_MODEL` | No | `gpt-4.1-nano` | Model for LLM + entity extraction |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama server (if using ollama) |
| `OLLAMA_MODEL` | No | `llama3.2` | Ollama model name |
