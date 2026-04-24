from fastapi import APIRouter
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rulebased import check_rule
from app.services.rag import search_docs
from app.services.llm_service import ask_llm

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    rule_answer, handoff = check_rule(request.message)

    if handoff:
        return ChatResponse(
            answer="Соединяю вас с оператором...",
            source="rule_based",
            handoff=True
        )

    if rule_answer:
        return ChatResponse(
            answer=rule_answer,
            source="rule_based",
            handoff=False
        )

    context = await search_docs(request.message)

    print(f"DEBUG — query: {request.message}")
    print(f"DEBUG — context length: {len(context)}")
    print(f"DEBUG — context preview: {context[:300] if context else 'EMPTY'}")

    if not context:
        # Нет документов — отвечаем без контекста
        answer = await ask_llm(request.message, "Контекст недоступен.", page=request.page, language=request.language)
    else:
        answer = await ask_llm(request.message, context, page=request.page, language=request.language)

    return ChatResponse(
        answer=answer,
        source="llm",
        handoff=False
    )