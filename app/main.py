from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routers.chat import router
from app.routers.documents import router as documents_router

app = FastAPI(
    title="iSEL AI Assistant",
    description="ИИ-ассистент платформы iSEL",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

app.include_router(router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "isel-bot"}