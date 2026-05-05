import os
import re
import uuid
import asyncio
import httpx
from pypdf import PdfReader
from docx import Document as DocxDocument
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "isel_docs")

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


async def get_embedding(text: str) -> list[float]:
    """Создаёт embedding через Ollama nomic-embed-text."""
    async with httpx.AsyncClient(timeout=60.0) as http:
        response = await http.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text}
        )
        return response.json()["embedding"]


def parse_pdf(filepath: str) -> str:
    """Извлекает текст из PDF."""
    reader = PdfReader(filepath)
    full_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            full_text += extracted + "\n"
    return full_text


def parse_docx(filepath: str) -> str:
    """Извлекает текст из DOCX, сохраняя структуру абзацев."""
    doc = DocxDocument(filepath)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


_KZ_CHARS = set("әғқңөұүіӘҒҚҢӨҰҮІ")


def _detect_language(text: str) -> str:
    return "kz" if any(ch in _KZ_CHARS for ch in text) else "ru"


def _extract_intent_name(text: str) -> str | None:
    m = re.search(r'Название интента:\s*([^\n]{5,})', text)
    if m:
        return m.group(1).strip()
    m = re.search(r'Название интента:\s*\n\s*([^\n]{5,})', text)
    if m:
        return m.group(1).strip()
    m = re.search(r'\d+\.\s*Интент:\s*([^\n]+)', text)
    return m.group(1).strip() if m else None


def _extract_page_context(text: str) -> str | None:
    m = re.search(r'(?:Страница|Бет)\s*[-–]\s*([^\n]+)|Контекст страницы:\s*([^\n]+)', text)
    if not m:
        return None
    return (m.group(1) or m.group(2)).strip()


def _slug_intent(intent_name: str | None, filename: str) -> tuple[str, str]:
    combined = ((intent_name or "") + " " + filename).lower()
    if "недвижим" in combined:
        return "real_estate", "Добавление объекта недвижимости"
    if any(kw in combined for kw in ["технические", "техусловия", "туслов"]):
        return "tu_application", "Заявление на технические условия"
    return "general", re.sub(r'\.[^.]+$', '', filename)


def _extract_situation_number(title: str) -> int | None:
    m = re.search(r'(?:Ситуация|Жағдай)\s+(\d+)', title)
    if m:
        return int(m.group(1))
    m = re.search(r'^(\d+)\.\d+', title.strip())
    return int(m.group(1)) if m else None


def split_by_situations(text: str, filename: str) -> list[dict]:
    """Splits document text into semantic situation chunks."""
    intent_slug, intent_display = _slug_intent(_extract_intent_name(text), filename)

    situation_pattern = r'(Ситуация\s+\d+[^\n]*|Жағдай\s+\d+[^\n]*|\d+\.\d+\s+[А-ЯЁ][^\n]*)'
    meta_pattern = r'(Цель интента|Описание интента|Когда активируется интент|Сценарии и ответы|Мақсаты|Сипаттамасы|Требования[^\n]*|Талаптар[^\n]*)'

    chunks = []

    meta_parts = re.split(meta_pattern, text)
    meta_chunks = []
    i = 0
    while i < len(meta_parts):
        part = meta_parts[i].strip()
        if re.match(meta_pattern, part):
            content = meta_parts[i + 1].strip() if i + 1 < len(meta_parts) else ""
            if content and len(content) > 50:
                meta_chunks.append({
                    "title": part,
                    "text": f"{part}\n\n{content}",
                    "situation": None,
                    "intent": intent_slug,
                    "intent_name": intent_display,
                })
            i += 2
        else:
            i += 1

    situation_blocks = re.split(situation_pattern, text)

    i = 0
    while i < len(situation_blocks):
        part = situation_blocks[i].strip()

        if re.match(r'(Ситуация\s+\d+|Жағдай\s+\d+|\d+\.\d+\s+[А-ЯЁ])', part):
            situation_title = part
            content = situation_blocks[i + 1].strip() if i + 1 < len(situation_blocks) else ""

            if content:
                has_page_marker = bool(re.search(r'(?:Страница|Бет)\s*[-–]|Контекст страницы:', content))
                if not has_page_marker and len(content) < 200:
                    i += 2
                    continue
                content_clean = re.sub(r'Шаблон ответа:?\s*|Жауап үлгісі:?\s*|Ответ бота:\s*', '', content).strip()
                full_chunk = f"{situation_title}\n\n{content_clean}"

                chunks.append({
                    "title": situation_title,
                    "text": full_chunk,
                    "situation": situation_title,
                    "intent": intent_slug,
                    "intent_name": intent_display,
                })
            i += 2
        else:
            i += 1

    chunks.extend(meta_chunks)

    seen: dict[str, int] = {}
    for i, chunk in enumerate(chunks):
        title = chunk["title"]
        match = re.match(r'(Ситуация\s+\d+|Жағдай\s+\d+|\d+\.\d+)', title)
        norm_key = match.group(1) if match else title[:60].strip()

        if norm_key not in seen:
            seen[norm_key] = i
        else:
            current_has_page = bool(re.search(r'(?:Страница|Бет)\s*[-–]|Контекст страницы:', chunk["text"]))
            existing_has_page = bool(re.search(r'(?:Страница|Бет)\s*[-–]|Контекст страницы:', chunks[seen[norm_key]]["text"]))
            if current_has_page and not existing_has_page:
                seen[norm_key] = i
            elif not current_has_page and not existing_has_page:
                if len(chunk["text"]) < len(chunks[seen[norm_key]]["text"]):
                    seen[norm_key] = i

    unique_indices = sorted(seen.values())
    chunks = [chunks[i] for i in unique_indices]

    filtered = []
    for chunk in chunks:
        text_content = chunk["text"]
        title = chunk["title"]

        if not any(h in title for h in ["Ситуация", "Жағдай"]) and not re.search(r'^\d+\.\d+', title):
            filtered.append(chunk)
            continue

        has_page = bool(re.search(r'(?:Страница|Бет)\s*[-–]|Контекст страницы:', text_content))
        has_steps = bool(re.search(r'\d+\.', text_content))
        has_bullets = "•" in text_content or "-" in text_content
        content_length = len(text_content) - len(title)

        if (has_page or has_steps or has_bullets) and content_length > 150:
            filtered.append(chunk)
        else:
            print(f"  🗑️  Filtered low-quality chunk: {title[:60]}")

    return filtered


def debug_chunks(chunks: list[dict], filename: str):
    """Print chunk contents to verify template is merged into situation."""
    print(f"\n=== DEBUG CHUNKS for {filename} ===")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1}: {chunk['title']} ---")
        print(chunk['text'][:500])
        print("...")
    print(f"=== END DEBUG ===\n")


def split_by_chunks(text: str, chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """Fallback word-based chunking with overlap."""
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
        i += chunk_size - overlap

    return chunks


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


def recreate_collection():
    """Drops and recreates the collection — wipes all existing vectors."""
    collections = client.get_collections().collections
    exists = any(c.name == QDRANT_COLLECTION for c in collections)

    if exists:
        client.delete_collection(QDRANT_COLLECTION)
        print(f"🗑️  Коллекция '{QDRANT_COLLECTION}' удалена.")

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    print(f"✅ Коллекция '{QDRANT_COLLECTION}' создана заново.")


async def ingest_file(filepath: str, filename: str):
    """Обрабатывает один файл и заливает чанки в Qdrant."""
    print(f"\n📄 Обработка: {filename}")

    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        text = parse_pdf(filepath)
    elif ext == "docx":
        text = parse_docx(filepath)
    elif ext in ["txt", "md"]:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print(f"⚠️  Формат .{ext} не поддерживается, пропускаем.")
        return

    if not text.strip():
        print(f"⚠️  Файл пустой, пропускаем.")
        return

    intent_name_raw = _extract_intent_name(text)
    intent_slug, intent_name = _slug_intent(intent_name_raw, filename)
    chunks = split_by_situations(text, filename)

    if not chunks:
        print(f"ℹ️  Смысловые блоки не найдены, используем разбивку по чанкам.")
        chunks = split_by_chunks(text)
        for chunk in chunks:
            chunk["intent"] = intent_slug
            chunk["intent_name"] = intent_name

    print(f"🎯 Intent: {intent_slug} ({intent_name})")
    print(f"📦 Найдено блоков: {len(chunks)}")
    debug_chunks(chunks, filename)

    for i, chunk in enumerate(chunks):
        vector = await get_embedding(chunk["text"])

        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}_{i}"))

        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "filename": filename,
                        "title": chunk["title"],
                        "text": chunk["text"],
                        "situation": chunk.get("situation"),
                        "chunk_idx": i,
                        "language": _detect_language(chunk["text"]),
                        "intent": chunk.get("intent", intent_slug),
                        "intent_name": chunk.get("intent_name", intent_name),
                        "situation_number": _extract_situation_number(chunk.get("title", "")),
                    }
                )
            ]
        )
        print(f"  ✅ [{i + 1}/{len(chunks)}] {chunk['title']}")


async def ingest():
    """Главная функция — заливает все файлы из data/docs/."""
    print("🚀 Запуск ingestion...\n")

    recreate_collection()

    doc_path = "data/docs/"
    supported = (".pdf", ".txt", ".md", ".docx")
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