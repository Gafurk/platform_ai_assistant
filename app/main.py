from fastapi import FastAPI
from app.routers.chat import router

app = FastAPI(
    title="iSEL AI Assistant",
    description="ИИ-ассистент платформы iSEL",
    version="0.1.0"
)

app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "isel-bot"}