import os
import shutil
import uuid
import asyncio
import json
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from app.utils.logger import (
    log_document_upload, log_ingestion_start,
    log_ingestion_complete, log_ingestion_error,
)

router = APIRouter()

DOCS_DIR = "data/docs"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Save document to data/docs/. Indexing happens on /reindex."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    os.makedirs(DOCS_DIR, exist_ok=True)
    safe_filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(DOCS_DIR, safe_filename)

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        file_size_kb = os.path.getsize(file_path) / 1024
        log_document_upload(safe_filename, round(file_size_kb, 1), "saved")

        return JSONResponse(content={
            "status": "saved",
            "filename": safe_filename,
            "message": "Файл сохранён. Нажмите «Переиндексировать» для обновления графа знаний.",
        })
    except Exception as e:
        log_document_upload(safe_filename, 0, f"failed: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/documents")
async def list_documents():
    """Return list of all uploaded documents."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    files = []

    for filename in os.listdir(DOCS_DIR):
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            filepath = os.path.join(DOCS_DIR, filename)
            files.append({
                "filename": filename,
                "size_kb": round(os.path.getsize(filepath) / 1024, 1),
                "extension": ext,
            })

    return {"documents": files, "total": len(files)}


@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Remove document from disk. Run /reindex to sync the knowledge graph."""
    filepath = os.path.join(DOCS_DIR, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    os.remove(filepath)
    return {"status": "deleted", "filename": filename}


@router.post("/reindex")
async def reindex():
    """Wipe LightRAG KG and reindex all docs in data/docs/. Streams SSE progress."""
    return StreamingResponse(
        _reindex_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


async def _reindex_stream():
    from app.services import lightrag_service
    from ingestion.ingest import ingest_file

    def evt(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    try:
        yield evt({"status": "wiping"})
        yield evt({"status": "init"})
        await lightrag_service.reinitialize()

        files = sorted([
            f for f in os.listdir(DOCS_DIR)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ])
        total = len(files)

        if total == 0:
            yield evt({"status": "done", "total": 0})
            return

        for i, filename in enumerate(files):
            yield evt({"status": "progress", "file": filename, "done": i, "total": total})
            try:
                log_ingestion_start(filename)
                result = await ingest_file(os.path.join(DOCS_DIR, filename), filename)
                chunks = result.get("chunks_count", 0) if isinstance(result, dict) else 0
                log_ingestion_complete(filename, chunks, "success")
            except Exception as e:
                log_ingestion_error(filename, str(e))
                yield evt({"status": "file_error", "file": filename, "error": str(e),
                           "done": i + 1, "total": total})
                continue
            await asyncio.sleep(0)

        yield evt({"status": "done", "total": total})

    except Exception as e:
        yield evt({"status": "error", "message": str(e)})
        yield evt({"status": "done"})
