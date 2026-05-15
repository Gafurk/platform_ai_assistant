from __future__ import annotations

from typing import Optional

from fastapi import APIRouter

from app.flows.base import FlowContext
from app.flows.faq import FAQFlow
from app.flows.registry import classify_intent, get_flow, has_flow_keywords
from app.models.schemas import ChatRequest, ChatResponse, FlowState
from app.services.llm_service import ask_llm
from app.services.rag import search_docs
from app.services.rate_limit import chat_limiter
from app.services.rulebased import check_rule
from app.services.session import sessions
from app.services.translit import normalize_kz
from app.services.validation import ValidationError, validate_message, validate_session_id
from app.utils.logger import log_chat_request

router = APIRouter()

# ---------------------------------------------------------------------------
# Constants — entity detection
# ---------------------------------------------------------------------------

_KZ_CHARS = frozenset("әғқңөұүіӘҒҚҢӨҰҮІ")

# Platform-specific entity abbreviations used only in Kazakh context.
# They contain no KZ-specific chars so history-walk can't help when the
# clarification gate fired (update_state_only skips history), so we
# treat them as an unconditional KZ signal.
_KZ_ENTITY_EXACT = frozenset({"жт", "зт", "ж.т.", "з.т."})

# Words that indicate the user wants to find/view/download rather than fill a form.
# Matched as substrings to handle inflected forms (статуса, документов, …).
_NAVIGATION_TRIGGERS = frozenset({
    # Russian
    "статус", "посмотреть", "где", "скачать", "документ",
    "готовое", "найти", "проверить",
    # Kazakh
    "мәртебе", "күй", "қайда", "қарау", "жүктеу", "дайын", "табу", "тексеру",
})

# Words that indicate form-filling — suppress the navigation shortcut when present.
# Includes the user-supplied list plus "ввести"/"указать" to prevent false positives
# on queries like "где ввести ИИН" or "где указать адрес".
_NAVIGATION_FILL_EXCLUSIONS = frozenset({
    "заполнить", "шаг", "подать", "создать",
    "ввести", "указать",
})

_FL_PHRASES = [
    "физическое лицо", "физ лицо", "физлицо",
    "физического лица", "физическим лицом",
    "жеке тұлға", "жеке тулга",
    "я фл", "как фл", "я ип",
]
_UL_PHRASES = [
    "юридическое лицо", "юр лицо", "юрлицо",
    "юридического лица", "юридическим лицом",
    "заңды тұлға", "занды тулга",
    "я юл", "как юл",
    " бин ", "наша организация", "наша компания",
]
_FL_EXACT = frozenset({"фл", "ф.л.", "физ", "физлицо", "физ лицо", "жт", "ж.т."})
_UL_EXACT = frozenset({"юл", "ю.л.", "юр", "юрлицо", "юр лицо", "зт", "з.т."})

# Steps that are shared between FL and UL — entity clarification not needed
_SHARED_STEPS = frozenset({"шаг 2", "шаг 3", "шаг 4", "шаг 5"})

# Russian function/pronoun words that strongly signal a Russian message.
# Matched whole-word (space-padded) to avoid false positives inside longer words.
_RU_FUNCTION_WORDS = frozenset({
    "где", "как", "что", "когда", "почему", "куда",
    "чей", "кто", "я", "мой", "подал",
})

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _safe_normalize_kz(message: str) -> str:
    """Apply Kazakh transliteration, reverting if KZ chars are introduced into a Russian message.

    normalize_kz maps some Russian-looking substrings (e.g. "тех условие") to
    proper Kazakh Cyrillic, which causes _detect_language to misidentify the
    message as Kazakh. Guard: if the original had no KZ chars but normalized
    has them AND the original contains Russian function words, revert.
    """
    if any(ch in _KZ_CHARS for ch in message):
        return message  # already proper KZ — nothing to transliterate
    normalized = normalize_kz(message)
    if any(ch in _KZ_CHARS for ch in normalized):
        padded = " " + message.lower() + " "
        if any((" " + w + " ") in padded for w in _RU_FUNCTION_WORDS):
            return message  # revert: normalization corrupted a Russian message
    return normalized


def _detect_language(message: str, history: list[dict] | None = None) -> str:
    if any(ch in _KZ_CHARS for ch in message):
        return "kz"
    # KZ entity abbreviations (ЖТ/ЗТ) have no KZ chars but are exclusively
    # used in Kazakh sessions. History walk can't help here because the
    # clarification gate uses update_state_only (no history entry saved yet).
    if message.strip().lower() in _KZ_ENTITY_EXACT:
        return "kz"
    # Short answers like "да", "иә" without KZ chars — inherit from history.
    if history and len(message.strip()) <= 10:
        for msg in reversed(history):
            if msg["role"] == "user" and len(msg["content"]) > 3:
                return "kz" if any(ch in _KZ_CHARS for ch in msg["content"]) else "ru"
    return "ru"


def _detect_entity(message: str, history: list[dict]) -> Optional[str]:
    stripped = message.strip().lower()
    if stripped in _FL_EXACT:
        return "физическое лицо"
    if stripped in _UL_EXACT:
        return "юридическое лицо"

    # Scan current message + all prior user turns
    all_text = " " + message.lower() + " "
    for msg in history:
        if msg["role"] == "user":
            all_text += " " + msg["content"].lower() + " "

    if any(ph in all_text for ph in _FL_PHRASES):
        return "физическое лицо"
    if any(ph in all_text for ph in _UL_PHRASES):
        return "юридическое лицо"
    return None


def _clarification_question(lang: str) -> str:
    if lang == "kz":
        return (
            "Нақты нұсқаулық беру үшін нақтылап кетіңізші: "
            "сіз өтінішті жеке тұлға (ЖТ) әлде заңды тұлға (ЗТ) ретінде бересіз бе?"
        )
    return (
        "Чтобы я дал точную инструкцию, подскажите: "
        "вы подаете заявление как физическое (ФЛ) или юридическое лицо (ЮЛ)?"
    )


def _needs_entity_clarification(
    intent: Optional[str], entity: Optional[str], page: Optional[str]
) -> bool:
    if entity is not None:
        return False
    # Only TU application branches on entity type (ФЛ/ЮЛ have different forms).
    # Real estate object addition is identical for all user types.
    if intent != "tu_application":
        return False
    # Shared steps (2–5) work the same for both entity types — skip gate
    if page and any(step in page.lower() for step in _SHARED_STEPS):
        return False
    return True


def _build_history_text(history: list[dict]) -> str:
    lines = []
    for msg in history[-6:]:
        role = "Пользователь" if msg["role"] == "user" else "Ассистент"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _annotate_question(message: str, entity: Optional[str], lang: str) -> str:
    if not entity:
        return message
    ann = (
        f"[пайдаланушы көрсетті: {entity}]" if lang == "kz"
        else f"[пользователь уже указал: {entity}]"
    )
    return f"{message} {ann}"


def _is_navigation_query(message: str) -> bool:
    """True if the message looks like a status/download/find query rather than form-filling.

    Fires when ANY navigation trigger is present and NO form-filling exclusion word
    is present. Substring matching handles inflected forms naturally.
    """
    lower = message.lower()
    if not any(t in lower for t in _NAVIGATION_TRIGGERS):
        return False
    return not any(e in lower for e in _NAVIGATION_FILL_EXCLUSIONS)


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # --- Input validation ---
    try:
        request.message = validate_message(request.message)
        request.session_id = validate_session_id(request.session_id)
    except ValidationError as e:
        return ChatResponse(
            answer=f"Неверный запрос: {str(e)}", source="validation", handoff=False
        )

    # --- Rate limiting ---
    if not chat_limiter.is_allowed(request.session_id):
        return ChatResponse(
            answer="Слишком много запросов. Подождите несколько секунд.",
            source="rate_limit",
            handoff=False,
        )

    sessions.cleanup()

    # --- Rule-based shortcut (greetings, identity, handoff) ---
    rule_answer, _ = check_rule(request.message)
    if rule_answer:
        log_chat_request(request.session_id, request.message, "rule_based")
        return ChatResponse(answer=rule_answer, source="rule_based", handoff=False)

    # --- Session load ---
    history = sessions.get_history(request.session_id)
    state = sessions.get_state(request.session_id)
    request.message = _safe_normalize_kz(request.message)
    print(f"DEBUG SAFE_NORM: {request.message!r}")
    print(f"DEBUG HAS_KZ: {any(ch in _KZ_CHARS for ch in request.message)}")
    lang = _detect_language(request.message, history)
    print(f"DEBUG LANG: {lang}")

    # --- Intent classification ---
    new_intent = classify_intent(request.message, request.page, state.intent)

    # --- Navigation FAQ shortcut ---
    # Intercepts status/download queries before the clarification gate and flow engine.
    # Handles both fresh sessions (avoids ФЛ/ЮЛ gate) and locked flows (catches flow-keyword
    # queries like "скачать ту" that the FAQ-interruption gate below would miss).
    # update_history_only preserves state so the user can resume form-filling afterwards.
    if _is_navigation_query(request.message):
        nav_context = await search_docs(request.message, intent=None, entity=None, language=lang)
        nav_answer = await ask_llm(
            request.message,
            nav_context if nav_context else "Контекст недоступен.",
            language=lang,
            history=_build_history_text(history),
            intent=None,
            entity=None,
            current_step=None,
            situation=None,
        )
        sessions.update_history_only(request.session_id, request.message, nav_answer)
        log_chat_request(request.session_id, request.message, "faq_interruption")
        return ChatResponse(answer=nav_answer, source="llm", handoff=False)

    # --- FAQ interruption: locked flow + no flow keywords + not a step phrase ---
    # Answer the FAQ question without disturbing the active flow state.
    active_flow = get_flow(state.intent)
    is_faq_interruption = (
        state.locked
        and state.intent is not None
        and not active_flow.is_step_progression(request.message)
        and not has_flow_keywords(request.message)
    )
    if is_faq_interruption:
        faq_ctx = FlowContext(
            message=request.message,
            language=lang,
            history=history,
            page=request.page,
            state=state,
        )
        query = FAQFlow().build_query(faq_ctx)
        context = await search_docs(query, intent=None, entity=None, language=lang)
        answer = await ask_llm(
            request.message,
            context if context else "Контекст недоступен.",
            language=lang,
            history=_build_history_text(history),
        )
        sessions.update_history_only(request.session_id, request.message, answer)
        log_chat_request(request.session_id, request.message, "faq_interruption")
        return ChatResponse(answer=answer, source="llm", handoff=False)

    # --- Intent update (only when not locked, or explicit switch) ---
    if new_intent != state.intent:
        if not state.locked:
            # Fresh intent — reset all flow state, capture original question
            state = FlowState(intent=new_intent, original_question=request.message)
        # If locked, explicit keyword switch is still allowed
        elif new_intent is not None and new_intent != state.intent:
            state = FlowState(intent=new_intent, original_question=request.message)

    # --- Entity detection ---
    if state.entity is None:
        detected = _detect_entity(request.message, history)
        if detected:
            state = state.model_copy(update={"entity": detected})

    # --- Entity clarification gate ---
    if _needs_entity_clarification(state.intent, state.entity, request.page):
        sessions.update_state_only(request.session_id, state)
        return ChatResponse(
            answer=_clarification_question(lang), source="rule_based", handoff=False
        )

    # --- Flow state transition ---
    flow = get_flow(state.intent)
    ctx = FlowContext(
        message=request.message,
        language=lang,
        history=history,
        page=request.page,
        state=state,
    )
    state = flow.next_state(ctx)

    # Lock once a flow is underway — prevents accidental intent resets.
    # TU: requires entity + step to be set (ФЛ/ЮЛ branches differ from step 1).
    # Real estate: locks as soon as the situation is chosen (no entity gate).
    _can_lock = (
        (state.intent == "tu_application" and state.entity is not None and state.step is not None)
        or (state.intent == "real_estate" and state.situation is not None)
    )
    if not state.locked and _can_lock:
        state = state.model_copy(update={"locked": True})

    # --- RAG retrieval ---
    ctx = ctx.__class__(
        message=request.message,
        language=lang,
        history=history,
        page=request.page,
        state=state,
    )
    query = flow.build_query(ctx)
    context = await search_docs(
        query,
        intent=state.intent,
        entity=state.entity,
        language=lang,
        current_step=state.step,
    )

    # --- LLM generation ---
    answer = await ask_llm(
        _annotate_question(request.message, state.entity, lang),
        context if context else "Контекст недоступен.",
        page=request.page,
        language=lang,
        history=_build_history_text(history),
        intent=state.intent,
        entity=state.entity,
        current_step=state.step,
        situation=state.situation,
    )

    sessions.update(request.session_id, state, request.message, answer)
    log_chat_request(request.session_id, request.message, "llm")
    return ChatResponse(answer=answer, source="llm", handoff=False)
