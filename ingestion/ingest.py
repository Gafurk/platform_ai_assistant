import os
import re
import uuid
import asyncio
import httpx
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from dotenv import load_dotenv

load_dotenv()

# Настройки
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "isel_docs")

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# ─── Embedding ────────────────────────────────────────────────────────────────

async def get_embedding(text: str) -> list[float]:
    """Создаёт embedding через Ollama nomic-embed-text."""
    async with httpx.AsyncClient(timeout=60.0) as http:
        response = await http.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text}
        )
        return response.json()["embedding"]


# ─── Парсинг PDF ──────────────────────────────────────────────────────────────

def parse_pdf(filepath: str) -> str:
    """Извлекает текст из PDF."""
    reader = PdfReader(filepath)
    full_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            full_text += extracted + "\n"
    return full_text


# ─── Смысловая разбивка ───────────────────────────────────────────────────────

def split_by_situations(text: str) -> list[dict]:
    """
    Splits document into semantic blocks by situation headers.
    Enriches each chunk with keywords for better RAG retrieval.
    """
    pattern = r'(Ситуация\s+\d+[^\n]*|Цель интента|Описание интента|Шаблон ответа|Требования[^\n]*)'
    parts = re.split(pattern, text)

    chunks = []
    i = 0
    current_situation = None

    while i < len(parts):
        part = parts[i].strip()

        if re.match(r'Ситуация\s+\d+', part):
            current_situation = part
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if content:
                chunks.append({
                    "title": part,
                    "text": f"{part}\n\n{content}",
                    "situation": part
                })
            i += 2

        elif part == 'Шаблон ответа':
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if content and current_situation:
                # Attach template directly to its situation
                # Find the last situation chunk and append template
                if chunks and chunks[-1].get("situation") == current_situation:
                    chunks[-1]["text"] += f"\n\nШаблон ответа:\n{content}"
                else:
                    chunks.append({
                        "title": f"Шаблон ответа — {current_situation}",
                        "text": f"{current_situation}\n\nШаблон ответа:\n{content}",
                        "situation": current_situation
                    })
            i += 2

        elif part in ['Цель интента', 'Описание интента'] or re.match(r'Требования', part):
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if content and len(content) > 50:
                chunks.append({
                    "title": part,
                    "text": f"{part}\n\n{content}",
                    "situation": None
                })
            i += 2

        else:
            if part and len(part) > 50:
                chunks.append({
                    "title": "Общая информация",
                    "text": part,
                    "situation": None
                })
            i += 1

    return chunks


def split_by_chunks(text: str, chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """
    Fallback — простая разбивка по словам с overlap.
    Используется если смысловая разбивка не нашла блоков.
    """
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "title": f"Блок {len(chunks) + 1}",
            "text": chunk_text
        })
        i += chunk_size - overlap  # overlap между чанками

    return chunks


# ─── Qdrant коллекция ─────────────────────────────────────────────────────────

def ensure_collection():
    """Создаёт коллекцию если не существует."""
    collections = client.get_collections().collections
    exists = any(c.name == QDRANT_COLLECTION for c in collections)

    if not exists:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        print(f"✅ Коллекция '{QDRANT_COLLECTION}' создана.")
    else:
        print(f"ℹ️  Коллекция '{QDRANT_COLLECTION}' уже существует.")


# ─── Основной ingestion ───────────────────────────────────────────────────────

async def ingest_file(filepath: str, filename: str):
    """Обрабатывает один файл и заливает чанки в Qdrant."""
    print(f"\n📄 Обработка: {filename}")

    # Парсинг
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        text = parse_pdf(filepath)
    elif ext in ["txt", "md"]:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print(f"⚠️  Формат .{ext} не поддерживается, пропускаем.")
        return

    if not text.strip():
        print(f"⚠️  Файл пустой, пропускаем.")
        return

    # Смысловая разбивка
    chunks = split_by_situations(text)

    # Fallback на простую разбивку если не нашли ситуации
    if not chunks:
        print(f"ℹ️  Смысловые блоки не найдены, используем разбивку по чанкам.")
        chunks = split_by_chunks(text)

    print(f"📦 Найдено блоков: {len(chunks)}")

    # Заливка в Qdrant
    for i, chunk in enumerate(chunks):
        vector = await get_embedding(chunk["text"])

        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "filename": filename,
                        "title": chunk["title"],
                        "text": chunk["text"],
                        "situation": chunk.get("situation"),
                        "chunk_idx": i
                    }
                )
            ]
        )
        print(f"  ✅ [{i + 1}/{len(chunks)}] {chunk['title']}")


async def ingest():
    """Главная функция — заливает все файлы из data/docs/."""
    print("🚀 Запуск ingestion...\n")

    # Проверяем/создаём коллекцию
    ensure_collection()

    doc_path = "data/docs/"
    supported = (".pdf", ".txt", ".md")
    files = [f for f in os.listdir(doc_path) if f.lower().endswith(supported)]

    if not files:
        print("⚠️  Файлы не найдены в data/docs/")
        return

    print(f"📁 Найдено файлов: {len(files)}")

    for filename in files:
        filepath = os.path.join(doc_path, filename)
        await ingest_file(filepath, filename)

    print("\n🎉 Ingestion завершён!")


if __name__ == "__main__":
    asyncio.run(ingest())