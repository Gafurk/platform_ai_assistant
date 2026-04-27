import os
import shutil
import uuid
import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

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

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run ingestion in background
    asyncio.create_task(_run_ingestion(file_path, safe_filename))

    return JSONResponse(content={
        "status": "uploaded",
        "filename": safe_filename,
        "message": "File uploaded. Ingestion started in background."
    })


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
    try:
        from ingestion.ingest import ingest_file
        await ingest_file(filepath, filename)
        print(f" Ingestion complete for: {filename}")
    except Exception as e:
        print(f" Ingestion failed for {filename}: {str(e)}")