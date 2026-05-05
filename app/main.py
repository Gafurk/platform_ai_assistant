from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routers.chat import router
from app.routers.documents import router as documents_router
from app.services import lightrag_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize LightRAG on startup."""
    await lightrag_service.initialize()
    yield


app = FastAPI(
    title="iSEL AI Assistant",
    description="ИИ-ассистент платформы iSEL",
    version="0.1.0",
    lifespan=lifespan,
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