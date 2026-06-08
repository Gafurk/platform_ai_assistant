import os
import re
import time
import asyncio
from pypdf import PdfReader
from docx import Document as DocxDocument
from dotenv import load_dotenv
from app.services import lightrag_service
from app.config.service_registry import ServiceRegistry

load_dotenv()




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
    """Detect service intent from document metadata and filename.

    Delegates to ServiceRegistry so adding new services requires only a
    YAML edit — no changes here.
    """
    registry = ServiceRegistry()
    text = intent_name or ""
    service_id = registry.detect_intent(text, filename)
    if service_id:
        display = registry.get_display_name(service_id, "ru")
        return service_id, display
    return "general", re.sub(r'\.[^.]+$', '', filename)


def _extract_situation_number(title: str) -> int | None:
    m = re.search(r'(?:Ситуация|Жағдай)\s+(\d+)', title)
    if m:
        return int(m.group(1))
    m = re.search(r'Шаг\s+(\d+)', title)
    if m:
        return int(m.group(1))
    m = re.search(r'^(\d+)\.\d+', title.strip())
    return int(m.group(1)) if m else None


def split_by_situations(text: str, filename: str) -> list[dict]:
    """Splits document text into semantic situation chunks."""
    intent_slug, intent_display = _slug_intent(_extract_intent_name(text), filename)

    situation_pattern = r'(Ситуация\s+\d+[^\n]*|Жағдай\s+\d+[^\n]*|Шаг\s+\d+[^\n]*|\d+\.\d+\s+[А-ЯЁ][^\n]*)'
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

        if re.match(r'(Ситуация\s+\d+|Жағдай\s+\d+|Шаг\s+\d+|\d+\.\d+\s+[А-ЯЁ])', part):
            situation_title = part
            content = situation_blocks[i + 1].strip() if i + 1 < len(situation_blocks) else ""

            if content:
                is_step_block = bool(re.match(r'Шаг\s+\d+', situation_title))
                has_page_marker = bool(re.search(r'(?:Страница|Бет)\s*[-–]|Контекст страницы:', content))
                if not is_step_block and not has_page_marker and len(content) < 200:
                    i += 2
                    continue
                content_clean = re.sub(r'Шаблон ответа:?\s*|Жауап үлгісі:?\s*|Ответ бота:\s*', '', content).strip()
                full_chunk = f"{situation_title}\n\n{content_clean}"

                # Refine intent from the situation title when the file-level
                # slug is "general" (e.g. "Общие вопросы" file containing
                # TU or real-estate situations).
                chunk_intent = intent_slug
                chunk_display = intent_display
                if intent_slug == "general":
                    refined, refined_display = _slug_intent(situation_title, filename)
                    if refined != "general":
                        chunk_intent = refined
                        chunk_display = refined_display

                chunks.append({
                    "title": situation_title,
                    "text": full_chunk,
                    "situation": situation_title,
                    "intent": chunk_intent,
                    "intent_name": chunk_display,
                })
            i += 2
        else:
            i += 1

    chunks.extend(meta_chunks)

    seen: dict[str, int] = {}
    for i, chunk in enumerate(chunks):
        title = chunk["title"]
        match = re.match(r'(Ситуация\s+\d+|Жағдай\s+\d+|Шаг\s+\d+|\d+\.\d+)', title)
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
            try:
                print(f"  Filtered low-quality chunk: {title[:60]}")
            except:
                pass

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


_ENTITY_MARKERS = re.compile(
    r'(Для физического лица\s*:?|Для юридического лица\s*:?|Для ФЛ\s*:?|Для ЮЛ\s*:?)',
    re.IGNORECASE,
)


def _split_by_entity(chunk: dict) -> list[dict]:
    """Split a step chunk into separate ФЛ/ЮЛ sub-chunks if both sections exist.

    Documents often contain both entity sections in one step:
        Шаг 1. "Данные заявителя"
        Для физического лица: ...
        Для юридического лица: ...

    This causes the LLM to reproduce both sections even when entity is known.
    Splitting ensures each chunk contains only one entity type.
    """
    parts = _ENTITY_MARKERS.split(chunk["text"])

    if len(parts) <= 1:
        return [chunk]

    result = []
    i = 1
    while i < len(parts) - 1:
        marker = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content and len(content) > 50:
            is_fl = "физическ" in marker.lower() or marker.upper().endswith("ФЛ:")
            entity_tag = "ФЛ" if is_fl else "ЮЛ"
            result.append({
                **chunk,
                "title": f"{chunk['title']} ({entity_tag})",
                "text": f"{chunk['title']}\n\n{marker}\n\n{content}",
            })
        i += 2

    return result if result else [chunk]


async def ingest_file(filepath: str, filename: str):
    """Process a document and insert structured blocks into LightRAG."""
    print(f"\n📄 Processing: {filename}")

    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        text = parse_pdf(filepath)
    elif ext == "docx":
        text = parse_docx(filepath)
    elif ext in ["txt", "md"]:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print(f"⚠️  Unsupported format .{ext}, skipping.")
        return

    if not text.strip():
        print(f"⚠️  Empty file, skipping.")
        return

    intent_name_raw = _extract_intent_name(text)
    intent_slug, intent_name = _slug_intent(intent_name_raw, filename)
    chunks = split_by_situations(text, filename)

    # Expand chunks that contain both ФЛ and ЮЛ sections into separate chunks.
    # Prevents LLM from reproducing both entity sections when only one is relevant.
    expanded = []
    for chunk in chunks:
        expanded.extend(_split_by_entity(chunk))
    if len(expanded) > len(chunks):
        print(f"🔀 Entity split: {len(chunks)} → {len(expanded)} chunks")
    chunks = expanded

    if not chunks:
        print(f"ℹ️  No semantic blocks found, using fixed-size chunking.")
        chunks = split_by_chunks(text)
        for chunk in chunks:
            chunk["intent"] = intent_slug
            chunk["intent_name"] = intent_name

    print(f"🎯 Intent: {intent_slug} ({intent_name})")
    print(f"📦 Found {len(chunks)} blocks")
    debug_chunks(chunks, filename)

    for i, chunk in enumerate(chunks):
        chunk_language = _detect_language(chunk["text"])
        metadata = f"[FILENAME: {filename}] [INTENT: {intent_slug}] [LANGUAGE: {chunk_language}] [TITLE: {chunk['title']}]"

        try:
            await lightrag_service.insert_text(chunk["text"], metadata=metadata)
            print(f"  ✅ [{i + 1}/{len(chunks)}] {chunk['title']}")
            if i < len(chunks) - 1:
                time.sleep(1)
        except Exception as e:
            print(f"  ❌ [{i + 1}/{len(chunks)}] {chunk['title']}: {str(e)}")
            raise

    return {"chunks_count": len(chunks)}


async def ingest():
    """Main ingestion function — processes all files from data/docs/ into LightRAG."""
    print("🚀 Starting LightRAG ingestion...\n")

    await lightrag_service.initialize()

    doc_path = "data/docs/"
    supported = (".pdf", ".txt", ".md", ".docx")
    files = [f for f in os.listdir(doc_path) if f.lower().endswith(supported)]

    if not files:
        print("⚠️  No files found in data/docs/")
        return

    print(f"📁 Found {len(files)} files")

    for filename in files:
        filepath = os.path.join(doc_path, filename)
        await ingest_file(filepath, filename)

    print("\n🎉 Ingestion complete!")


if __name__ == "__main__":
    asyncio.run(ingest())