from fastapi import APIRouter
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rulebased import check_rule
from app.services.rag import search_docs
from app.services.llm_service import ask_llm

router = APIRouter()

sessions: dict[str, list[dict]] = {}

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

    history = sessions.get(request.session_id, [])

    if history and ("физическое лицо" in request.message.lower() or "юридическое лицо" in request.message.lower()):
        last_user = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"),
            ""
        )
        enriched_query = f"{last_user} {request.message}"
    else:
        enriched_query = request.message

    context = await search_docs(enriched_query)

    print(f"DEBUG — query: {enriched_query}")
    print(f"DEBUG — context length: {len(context)}")
    print(f"DEBUG — context preview: {context[:300] if context else 'EMPTY'}")

    history_text = ""
    if history:
        for msg in history[-4:]:
            role = "Пользователь" if msg["role"] == "user" else "Ассистент"
            history_text += f"{role}: {msg['content']}\n"

    if not context:
        answer = await ask_llm(request.message, "Контекст недоступен.", page=request.page, language=request.language, history=history_text)
    else:
        answer = await ask_llm(request.message, context, page=request.page, language=request.language, history=history_text)

    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": answer})
    sessions[request.session_id] = history[-10:]

    return ChatResponse(
        answer=answer,
        source="llm",
        handoff=False
    )