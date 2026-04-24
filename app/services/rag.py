import httpx
import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "isel_docs")

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Create embedding through Ollama nomic-embed-text model
async def get_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=60.0) as http:
        response = await http.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text}
        )
        return response.json()["embedding"]

# Search in Qdrant using the embedding and return context
async def search_docs(query: str, top_k: int = 5) -> str:
    try:
        embedding = await get_embedding(query)

        results = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=embedding,
            limit=top_k
        )

        points = results.points

        if not points:
            return ""

        # Prioritize situation templates over general info
        priority_points = [
            p for p in points
            if any(kw in p.payload.get('title', '')
                   for kw in ['Ситуация', 'Шаблон ответа'])
        ]
        other_points = [
            p for p in points
            if p not in priority_points
        ]
        sorted_points = priority_points + other_points

        context = "\n\n".join([
            f"[{p.payload.get('title', 'doc')}]:\n{p.payload.get('text', '')}"
            for p in sorted_points
        ])
        return context

    except Exception as e:
        print(f"Qdrant search error: {e}")
        return ""