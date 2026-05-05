import httpx
import os
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
from app.utils.logger import log_rag_search, log_qdrant_error

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "isel_docs")

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


async def _retry_with_backoff(coro_fn, max_retries: int = 3, base_delay: float = 1.0):
    """Retry async operation with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await coro_fn()
        except (httpx.TimeoutException, httpx.ConnectError, Exception) as e:
            if attempt == max_retries - 1:
                print(f"❌ Retry failed after {max_retries} attempts: {str(e)}")
                raise
            delay = base_delay * (2 ** attempt)
            print(f"⚠️  Attempt {attempt + 1} failed: {str(e)}, retrying in {delay}s...")
            await asyncio.sleep(delay)


async def get_embedding(text: str) -> list[float]:
    async def _fetch():
        async with httpx.AsyncClient(timeout=60.0) as http:
            response = await http.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": OLLAMA_EMBED_MODEL, "prompt": text}
            )
            response.raise_for_status()
            return response.json()["embedding"]

    return await _retry_with_backoff(_fetch, max_retries=3, base_delay=1.0)


async def search_docs(query: str, top_k: int = 7, intent: str = None, entity: str = None, page: str = None, language: str = None) -> str:
    try:
        expanded = query
        lower = query.lower()

        if intent in ("tu_application", None):
            if "ту " in lower or lower.endswith("ту") or "тқ" in lower:
                expanded = f"{query} заявление на выдачу технических условий техникалық шарттар"

        if intent != "tu_application":
            if "он " in lower or "объект" in lower or "недвижим" in lower:
                expanded = f"{query} добавление объекта недвижимости"

        print(f"DEBUG — RAG intent={intent} expanded query: {expanded}")

        embedding = await get_embedding(expanded)

        filters = []
        if intent:
            filters.append(FieldCondition(key="intent", match=MatchValue(value=intent)))
        if language:
            filters.append(FieldCondition(key="language", match=MatchValue(value=language)))

        qdrant_filter = Filter(must=filters) if filters else None

        results = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=embedding,
            query_filter=qdrant_filter,
            limit=top_k,
        )

        points = results.points

        if not points and qdrant_filter is not None:
            results = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=embedding,
                limit=top_k,
            )
            points = results.points

        if not points:
            return ""

        priority_points = [
            p for p in points
            if any(kw in p.payload.get('title', '')
                   for kw in ['Ситуация', 'Шаблон ответа', 'Жағдай', 'Жауап үлгісі'])
        ]
        other_points = [p for p in points if p not in priority_points]
        sorted_points = priority_points + other_points

        context_parts = []
        for p in sorted_points:
            title = p.payload.get('title', '')
            text = p.payload.get('text', '')
            filename = p.payload.get('filename', '')
            context_parts.append(
                f"=== ДОКУМЕНТ: {filename} | РАЗДЕЛ: {title} ===\n{text}"
            )

        context = "\n\n".join(context_parts)

        print(f"DEBUG — context length: {len(context)}")
        print(f"DEBUG — context preview: {context[:1000] if context else 'EMPTY'}")
        print(f"DEBUG — number of chunks found: {len(points)}")
        for idx, p in enumerate(sorted_points[:3]):
            print(f"DEBUG — chunk {idx+1} title: {p.payload.get('title', '')}")
            print(f"DEBUG — chunk {idx+1} text preview: {p.payload.get('text', '')[:200]}")

        log_rag_search(query, intent or "none", language or "auto", len(points))
        return context

    except Exception as e:
        log_qdrant_error(f"{type(e).__name__}: {str(e)}")
        print(f"❌ RAG search failed: {type(e).__name__}: {str(e)}")
        return ""