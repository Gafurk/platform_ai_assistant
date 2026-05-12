from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter

from app.models.schemas import ChatRequest, ChatResponse, SessionState
from app.services.rulebased import check_rule
from app.services.rag import search_docs
from app.services.llm_service import ask_llm
from app.services.validation import validate_message, validate_session_id, ValidationError
from app.services.rate_limit import chat_limiter
from app.utils.logger import log_chat_request, log_chat_error

router = APIRouter()

sessions: dict[str, list[dict]] = {}
session_states: dict[str, SessionState] = {}
session_accessed: dict[str, float] = {}

SESSION_TIMEOUT_SECONDS = 86400  # 24 hours


def _cleanup_expired_sessions():
    """Remove sessions idle for more than SESSION_TIMEOUT_SECONDS."""
    now = time.time()
    expired = [sid for sid, accessed_at in session_accessed.items() if now - accessed_at > SESSION_TIMEOUT_SECONDS]
    for sid in expired:
        sessions.pop(sid, None)
        session_states.pop(sid, None)
        session_accessed.pop(sid, None)

_FL_PHRASES = [
    "физическое лицо", "физ лицо", "физлицо",
    "физического лица", "физическим лицом",
    "жеке тұлға", "жеке тулга",
    "я фл", "как фл", "я ип",
    "физ лицо",
]
_UL_PHRASES = [
    "юридическое лицо", "юр лицо", "юрлицо",
    "юридического лица", "юридическим лицом",
    "заңды тұлға", "занды тулга",
    "я юл", "как юл",
    " бин ", "наша организация", "наша компания",
    "юр лицо",
]

_FL_EXACT = {"фл", "ф.л.", "физ", "физлицо", "физ лицо", "жт", "ж.т."}
_UL_EXACT = {"юл", "ю.л.", "юр", "юрлицо", "юр лицо", "зт", "з.т."}

_TU_KEYWORDS = [" ту ", "технические условия", " тқ ", "техникалық шарттар", "техусловия"]
_RE_KEYWORDS = ["объект недвижимости", "добавить объект", "добавление объекта", "кадастр"]

_TU_PAGE_KEYWORDS = ["технических условий", "техусловия", "шаг 1", "шаг 2", "шаг 3", "шаг 4", "шаг 5"]
_RE_PAGE_KEYWORDS = ["объект недвижимости", "объекты недвижимости", "добавление объекта"]

_SHARED_STEPS = ["шаг 2", "шаг 3", "шаг 4", "шаг 5"]

_TU_MAX_STEPS = 5
_NEXT_STEP_PHRASES = {
    "дальше", "далее", "следующий", "следующий шаг",
    "продолжай", "продолжи", "давай", "да",
    "келесі", "жалғастыр", "ары қарай",
}

def _clarification_question(lang: str) -> str:
    if lang == "kz":
        return "Нақты нұсқаулық беру үшін нақтылап кетіңізші: сіз өтінішті жеке тұлға (ЖТ) әлде заңды тұлға (ЗТ) ретінде бересіз бе?"
    return "Чтобы я дал точную инструкцию, подскажите: вы подаете заявление как физическое (ФЛ) или юридическое лицо (ЮЛ)?"


def _detect_entity(message: str, history: list[dict]) -> Optional[str]:
    stripped = message.strip().lower()
    if stripped in _FL_EXACT:
        return "физическое лицо"
    if stripped in _UL_EXACT:
        return "юридическое лицо"
    all_user_text = " " + message.lower() + " "
    for msg in history:
        if msg["role"] == "user":
            all_user_text += " " + msg["content"].lower() + " "

    if any(ph in all_user_text for ph in _FL_PHRASES):
        return "физическое лицо"
    if any(ph in all_user_text for ph in _UL_PHRASES):
        return "юридическое лицо"
    return None


def _detect_intent(message: str) -> Optional[str]:
    """Keyword-based intent detection from the current message."""
    lower = " " + message.lower() + " "
    if any(kw in lower for kw in _TU_KEYWORDS):
        return "tu_application"
    if any(kw in lower for kw in _RE_KEYWORDS):
        return "real_estate"
    return None


def _classify_intent(message: str, page: Optional[str], state: SessionState) -> Optional[str]:
    lower_msg = " " + message.lower() + " "
    if state.intent is not None:
        if any(kw in lower_msg for kw in _RE_KEYWORDS) and state.intent == "tu_application":
            return "real_estate"
        if any(kw in lower_msg for kw in _TU_KEYWORDS) and state.intent == "real_estate":
            return "tu_application"
        return state.intent
    if page:
        lower_page = page.lower()
        if any(kw in lower_page for kw in _TU_PAGE_KEYWORDS):
            return "tu_application"
        if any(kw in lower_page for kw in _RE_PAGE_KEYWORDS):
            return "real_estate"
    return _detect_intent(message)


def _needs_entity_clarification(intent: Optional[str], entity: Optional[str], page: Optional[str]) -> bool:
    """Returns True when the router should ask for FL/UL before any RAG or LLM call."""
    if entity is not None:
        return False
    if intent not in ("real_estate", "tu_application"):
        return False
    if page:
        lower_page = page.lower()
        if any(step in lower_page for step in _SHARED_STEPS):
            return False
    return True


def _detect_language(message: str) -> str:
    kz_chars = set("әғқңөұүіӘҒҚҢӨҰҮІ")
    if any(ch in kz_chars for ch in message):
        return "kz"
    return "ru"


def _is_next_step_request(message: str) -> bool:
    """Check if user is asking for the next step."""
    return message.strip().lower() in _NEXT_STEP_PHRASES


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        request.message = validate_message(request.message)
        request.session_id = validate_session_id(request.session_id)
    except ValidationError as e:
        return ChatResponse(answer=f"❌ Invalid input: {str(e)}", source="validation", handoff=False)

    if not chat_limiter.is_allowed(request.session_id):
        return ChatResponse(
            answer="⏱️ Слишком много запросов. Подождите несколько секунд перед следующим сообщением.",
            source="rate_limit",
            handoff=False
        )

    _cleanup_expired_sessions()
    session_accessed[request.session_id] = time.time()

    rule_answer, handoff = check_rule(request.message)

    if handoff:
        log_chat_request(request.session_id, request.message, "handoff")
        return ChatResponse(
            answer="Соединяю вас с оператором...",
            source="rule_based",
            handoff=True
        )

    if rule_answer:
        log_chat_request(request.session_id, request.message, "rule_based")
        return ChatResponse(
            answer=rule_answer,
            source="rule_based",
            handoff=False
        )

    history = sessions.get(request.session_id, [])
    state = session_states.setdefault(request.session_id, SessionState())

    detected_lang = _detect_language(request.message)

    previous_intent = state.intent
    previous_entity = state.entity
    state.intent = _classify_intent(request.message, request.page, state)
    if state.intent is not None and state.intent != previous_intent:
        state.entity = None
        state.step = None

    if state.entity is None:
        state.entity = _detect_entity(request.message, history)

    entity = state.entity

    if _needs_entity_clarification(state.intent, entity, request.page):
        return ChatResponse(answer=_clarification_question(detected_lang), source="rule_based", handoff=False)

    if state.intent == "tu_application":
        is_entity_fix = (state.entity is not None) and (previous_entity is None)
        is_next = _is_next_step_request(request.message) and state.step is not None

        if is_entity_fix and state.step is None:
            old_step = state.step
            state.step = 1
            print(f"[State] Entity fixed → Step {old_step} -> 1")

        elif is_next:
            old_step = state.step
            state.step = min(state.step + 1, _TU_MAX_STEPS)
            if old_step != state.step:
                print(f"[State] Step transition: {old_step} -> {state.step}")
            else:
                print(f"[State] Step capped at max: {_TU_MAX_STEPS}")

        elif state.step is None and state.entity is not None:
            state.step = 1
            print(f"[State] TU intent initialized → Step None -> 1")
    else:
        state.step = None

    if entity and history:
        last_user = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"), ""
        )
        enriched_query = f"{last_user} {request.message}" if last_user else request.message
    elif entity:
        enriched_query = f"{request.message} {entity}"
    else:
        enriched_query = request.message

    if entity:
        annotation = (
            f"[пайдаланушы көрсетті: {entity}]" if detected_lang == "kz"
            else f"[пользователь уже указал: {entity}]"
        )
        question_for_llm = f"{request.message} {annotation}"
    else:
        question_for_llm = request.message

    print(f"DEBUG — intent: {state.intent}, entity: {entity}, step: {state.step}")
    print(f"DEBUG — detected_lang: {detected_lang}")
    print(f"DEBUG — enriched_query: {enriched_query}")

    context = await search_docs(enriched_query, intent=state.intent, entity=entity, language=detected_lang, current_step=state.step)

    history_text = ""
    if history:
        for msg in history[-6:]:
            role = "Пользователь" if msg["role"] == "user" else "Ассистент"
            history_text += f"{role}: {msg['content']}\n"

    context_str = context if context else "Контекст недоступен."
    answer = await ask_llm(
        question_for_llm, context_str,
        page=request.page, language=detected_lang, history=history_text,
        intent=state.intent, entity=entity, current_step=state.step,
    )

    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": answer})
    sessions[request.session_id] = history[-20:]

    log_chat_request(request.session_id, request.message, "llm")
    return ChatResponse(answer=answer, source="llm", handoff=False)