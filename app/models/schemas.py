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


class FlowState(BaseModel):
    """Per-session flow state persisted in SessionManager."""
    intent: Optional[str] = None
    entity: Optional[str] = None
    step: Optional[int] = None       # linear flows (TU)
    situation: Optional[str] = None  # scenario flows (Real Estate)
    locked: bool = False             # True once a flow is underway
    original_question: Optional[str] = None  # first user message that triggered this intent


# Backward-compat alias — remove after all callers updated
SessionState = FlowState
