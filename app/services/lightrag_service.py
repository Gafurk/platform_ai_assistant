import os
import numpy as np
from openai import AsyncOpenAI
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from app.utils.logger import log_rag_search, log_qdrant_error

# Separate clients with different timeouts
_oai_embeddings = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""), timeout=300.0)
_oai_llm = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""), timeout=600.0)


async def gpt41_nano_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] = [],
    **kwargs,
) -> str:
    """Custom LLM function for gpt-4.1-nano via OpenAI SDK."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    try:
        response = await _oai_llm.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 2000),
        )
        return response.choices[0].message.content
    except Exception as e:
        log_qdrant_error(f"gpt41_nano_complete failed: {str(e)}")
        raise


async def openai_embed(texts: list[str]) -> np.ndarray:
    """Embedding function using OpenAI text-embedding-3-small."""
    try:
        response = await _oai_embeddings.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
            encoding_format="float",
        )
        return np.array([item.embedding for item in response.data])
    except Exception as e:
        log_qdrant_error(f"openai_embed failed: {str(e)}")
        raise


# Initialize LightRAG with GPT-4.1-nano and OpenAI embeddings
# Timeouts are configured at the OpenAI client level (_oai_embeddings, _oai_llm)
rag = LightRAG(
    working_dir="./data/lightrag",
    llm_model_func=gpt41_nano_complete,
    embedding_func=EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=openai_embed,
    ),
    chunk_token_size=1200,
    chunk_overlap_token_size=100,
    rerank_model_func=None,
)


# ---------------------------------------------------------------------------
# Context post-filtering
# ---------------------------------------------------------------------------

# Keywords that identify chunks as belonging to a specific service intent.
# Both RU and KZ forms are included because the KG stores bilingual content.
_INTENT_FILTER_KEYWORDS: dict[str, list[str]] = {
    "tu_application": [
        # Russian
        "технические условия", "техусловия", "заявление на технические",
        "шаг 1", "шаг 2", "шаг 3", "шаг 4", "шаг 5",
        "фио", "иин", "бин", "физическое лицо", "юридическое лицо",
        "электроснабжение", "электрических сетей",
        # Kazakh
        "техникалық шарттар", "жеке тұлға", "заңды тұлға",
        "1-қадам", "2-қадам", "3-қадам", "4-қадам", "5-қадам",
        "электрмен жабдықтау",
    ],
    "real_estate": [
        # Russian
        "объект недвижимости", "кадастровый номер",
        "адресный регистр", "добавление объекта", "через кадастр",
        # Kazakh
        "жылжымайтын мүлік", "кадастрлық нөмір",
        "адрестік регистр", "объект қосу",
    ],
    "supply_contract_non_residential": [
        # Russian
        "договор небытов", "акцепт договора", "заключение договора",
        "встроенное помещение", "пристроенное помещение",
        # Kazakh
        "тұрмыстық емес шарт", "электрмен жабдықтау шарты",
    ],
    "supply_contract_residential": [
        # Russian
        "договор бытов", "договор электроснабжения",
        "акцепт договора", "заключение договора",
        "кондоминиум", "количество проживающих",
        # Kazakh
        "тұрмыстық шарт", "электрмен жабдықтау шарты",
    ],
    "load_calculation": [
        # Russian
        "расчет нагрузки", "расчёт нагрузки", "электрическая нагрузка",
        "расчет электрической",
        # Russian — unique field names absent from other services
        "вид объекта недвижимости", "количество квартир",
        "уровень электрификации", "электроприемник", "целевое назначение",
        # Kazakh
        "жүктеме есебі", "жүктемені есептеу", "электр жүктемесі",
        "жүктемесін", "жүктемені", "жүктемесі",
    ],
    "draft_design": [
        # Russian
        "эскизный проект", "разработка эскизного", "проект внешнего",
        # Russian — unique field names absent from other services
        "тип подключения", "расстояние до точки", "топографическая съемка",
        "проект внешнего электроснабжения",
        # Kazakh
        "эскиздік жоба", "эскиздік жобаны әзірлеу",
    ],
    "construction_works": [
        # Russian
        "строительно-монтажные", "строительно монтажные", " смр ",
        "монтажные работы", "строительные работы",
        # Russian — unique field names absent from other services
        "очередь строительства", "год реализации", "этап строительства",
        # Kazakh
        "құрылыс-монтаж", "монтаждау жұмыстары", "құрылыс монтаждау жұмыстары",
    ],
    "meter_sealing": [
        # Russian
        "пломб", "установка пломбы", "снятие пломбы", "прибор учета",
        # Russian — unique field names absent from other services
        "заводской номер", "лицевой счет", "абонентский номер",
        "причина установки", "причина снятия",
        # Kazakh
        "пломбаны орнату", "пломбаны алу", "есептеуіш аспап",
    ],
}


def _filter_context_by_intent(context: str, intent: str | None) -> str:
    """Remove chunks from unrelated services to avoid mixed templates.

    Splits the LightRAG context on double-newline paragraph breaks and keeps
    only chunks that contain intent-specific keywords or the [INTENT:] tag.
    Falls back to the full context if filtering removes everything.
    """
    if not intent or not context:
        return context
    keywords = _INTENT_FILTER_KEYWORDS.get(intent, [])
    if not keywords:
        return context

    chunks = context.split("\n\n")
    relevant = [
        chunk for chunk in chunks
        if any(kw in chunk.lower() for kw in keywords)
        or f"[INTENT: {intent}]" in chunk
    ]
    # If filtering removed everything, return original — mixed is better than empty
    return "\n\n".join(relevant) if relevant else context


async def initialize():
    """Initialize LightRAG storage. Must be called in FastAPI lifespan."""
    try:
        await rag.initialize_storages()
        print("✅ LightRAG initialized")
    except Exception as e:
        print(f"❌ LightRAG initialization failed: {str(e)}")
        raise


async def search_docs(
    query: str,
    top_k: int = 7,
    intent: str = None,
    entity: str = None,
    page: str = None,
    language: str = None,
    current_step: int = None,
) -> str:
    """
    Search the knowledge graph for context.

    Args:
        query: User question
        top_k: Number of nodes to retrieve (passed to QueryParam)
        intent: "tu_application" or "real_estate" (prepended to query)
        entity: "физическое лицо" or "юридическое лицо" (prepended to query)
        page: Page context (not directly used, kept for API compatibility)
        language: "ru" or "kz" (kept for future language-specific filtering)
        current_step: Current step number for multi-step processes (prepended to query)

    Returns:
        Formatted context string for the LLM
    """
    try:
        # Prepend step, intent, and entity to query for graph extraction
        enriched_query = query
        if current_step is not None:
            enriched_query = f"[Шаг {current_step}] {enriched_query}"
        if intent:
            enriched_query = f"[{intent}] {enriched_query}"
        if entity:
            enriched_query = f"[{entity}] {enriched_query}"
            # Append entity-specific terms to help LightRAG disambiguate
            # ФЛ vs ЮЛ chunks when both exist for the same step.
            if "юридическое" in entity:
                enriched_query += " БИН организация руководитель"
            elif "физическое" in entity:
                enriched_query += " ФИО ИИН"

        print(f"DEBUG — LightRAG query: step={current_step}, intent={intent}, entity={entity}, lang={language}, q_len={len(query)}")

        # Query knowledge graph with only_need_context=True
        # This returns raw context without LLM generation
        result = await rag.aquery(
            enriched_query,
            param=QueryParam(
                mode="hybrid",
                only_need_context=True,
                top_k=top_k,
            ),
        )

        # LightRAG returns a string with context
        context = result if isinstance(result, str) else str(result)

        # Filter out chunks from unrelated services to avoid mixed templates
        if intent:
            context = _filter_context_by_intent(context, intent)

        log_rag_search(query, intent or "none", language or "auto", len(context.split("\n")))

        print(f"DEBUG — LightRAG context_len={len(context)}, preview={context[:300] if context else 'EMPTY'}")

        if intent and len(context) < 100:
            print(f"⚠️  LightRAG returned very little context for intent={intent}. Consider lowering filter strictness.")

        return context if context else ""

    except Exception as e:
        log_qdrant_error(f"LightRAG search failed: {type(e).__name__}: {str(e)}")
        print(f"❌ LightRAG search failed: {type(e).__name__}: {str(e)}")
        return ""


async def insert_text(text: str, metadata: str = ""):
    """
    Insert text into the knowledge graph.

    Args:
        text: Document text to insert
        metadata: Optional metadata string (e.g., "[INTENT: tu_application] [LANGUAGE: ru]")
    """
    if not text or not text.strip():
        print(f"⚠️  Skipping empty text insert")
        return

    insert_text_content = text
    if metadata:
        insert_text_content = f"{metadata}\n\n{text}"

    try:
        await rag.ainsert(insert_text_content)
        print(f"✅ Inserted text ({len(text)} chars) into knowledge graph")
    except Exception as e:
        print(f"❌ LightRAG insert failed: {str(e)}")
        raise
