import os
import shutil
import uuid
import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.services.task_queue import ingestion_queue
from app.utils.logger import log_document_upload, log_ingestion_start, log_ingestion_complete, log_ingestion_error

router = APIRouter()

DOCS_DIR = "data/docs"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document and trigger ingestion into Qdrant."""

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
        log_document_upload(safe_filename, round(file_size_kb, 1), "uploaded")

        # Queue ingestion with concurrency limit
        await ingestion_queue.submit(_run_ingestion(file_path, safe_filename))

        queue_status = await ingestion_queue.get_status()
        return JSONResponse(content={
            "status": "queued",
            "filename": safe_filename,
            "message": f"File queued for ingestion. Active: {queue_status['active_tasks']}, Queued: {queue_status['queued_tasks']}"
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
                "extension": ext
            })

    return {"documents": files, "total": len(files)}


@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document from disk."""
    filepath = os.path.join(DOCS_DIR, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    os.remove(filepath)
    return {"status": "deleted", "filename": filename}


async def _run_ingestion(filepath: str, filename: str):
    """Run ingestion for a single file."""
    log_ingestion_start(filename)
    try:
        from ingestion.ingest import ingest_file
        result = await ingest_file(filepath, filename)
        chunks_count = result.get("chunks_count", 0) if isinstance(result, dict) else 0
        log_ingestion_complete(filename, chunks_count, "success")
        print(f"✅ Ingestion complete for: {filename}")
    except Exception as e:
        log_ingestion_error(filename, str(e))
        print(f"❌ Ingestion failed for {filename}: {str(e)}")