from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    message: str
    session_id: str
    page: Optional[str] = None
    language: Optional[str] = "ru"


class ChatResponse(BaseModel):
    answer: str
    source: str
    handoff: bool = False


class SessionState(BaseModel):
    intent: Optional[str] = None
    entity: Optional[str] = None
    step: Optional[int] = None
    situation: Optional[int] = None