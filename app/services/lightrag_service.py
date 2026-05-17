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
# Context post-filtering — metadata tag based (Phase 3)
# ---------------------------------------------------------------------------

def _filter_context_by_intent(
    context: str, intent: str | None, current_step: int | None = None
) -> str:
    """Keep only chunks whose [INTENT:] metadata tag matches the current intent.

    Each ingested chunk starts with:
        [FILENAME: ...] [INTENT: {slug}] [LANGUAGE: ...] [TITLE: ...]

    LightRAG returns chunks as JSON objects separated by newlines. We split on
    the JSON object boundary so each element is a complete chunk (metadata + body),
    not a paragraph fragment.

    When current_step is provided, after intent-filtering we further prefer chunks
    whose text contains "Шаг N" — this prevents terminology/FAQ blocks that share
    the same intent tag from being returned for the wrong step.

    When LightRAG returns entity-only context (0 vector chunks, no reference_id
    objects), there are no [INTENT:] tags to match. In that case we pass the full
    context through so the LLM can use entity descriptions instead of getting an
    empty context and falling back to navigation hints.
    """
    import re as _re

    if not intent or not context:
        return context

    tag = f"[INTENT: {intent}]"

    # Split on JSON chunk boundaries: each chunk is a {"reference_id":...} object.
    # Keep the header portion (entity section, markdown fences) intact.
    parts = _re.split(r'(?=\{"reference_id")', context)
    header = ""
    chunks: list[str] = []
    for part in parts:
        if part.lstrip().startswith('{"reference_id"'):
            chunks.append(part)
        else:
            header += part

    if chunks:
        relevant = [c for c in chunks if tag in c]
        # When step is known, prefer chunks that contain "Шаг N" — this filters out
        # same-intent terminology or FAQ blocks that share the intent tag but belong
        # to a different section (e.g. "Ситуация 4" terminology vs "Шаг 4" form).
        if relevant and current_step is not None:
            step_tag = f"Шаг {current_step}"
            step_relevant = [c for c in relevant if step_tag in c]
            if step_relevant:
                relevant = step_relevant
        if relevant:
            return header + "".join(relevant)
        # Chunks found but none match intent — safer than returning unrelated service content.
        return ""

    # Fallback: non-JSON context (legacy test format or entity-only LightRAG output).
    paragraphs = [c for c in context.split("\n\n") if c.strip()]
    relevant = [c for c in paragraphs if tag in c]
    if relevant:
        return "\n\n".join(relevant)
    # If no paragraph carries any [INTENT:] tag at all, this is entity-only context
    # (LightRAG returned 0 vector chunks). Pass it through so the LLM can use entity
    # descriptions rather than receiving an empty context.
    if not any("[INTENT:" in c for c in paragraphs):
        return context
    return ""


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
    top_k: int = 20,
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
            # "[TITLE: Шаг N]" matches the exact metadata prefix in ingested chunks,
            # improving vector similarity for the right step.
            enriched_query = f"[TITLE: Шаг {current_step}] {enriched_query}"
        if intent:
            # "[INTENT: ...]" matches the exact metadata prefix in ingested chunks,
            # so the vector search prefers chunks from the correct service.
            enriched_query = f"[INTENT: {intent}] {enriched_query}"
        if entity:
            enriched_query = f"[{entity}] {enriched_query}"
            # NOTE: Do NOT append "ФИО ИИН" / "БИН организация руководитель" here.
            # Those suffixes bias the entity graph toward Step-1 entity chunks for
            # every step query (ФИО/ИИН appear in many documents and dominate graph
            # traversal, causing Steps 2-5 to return Step-1 content).

        print(f"DEBUG — LightRAG query: step={current_step}, intent={intent}, entity={entity}, lang={language}, q_len={len(query)}")

        # Primary search: hybrid mode (graph traversal + vector similarity).
        result = await rag.aquery(
            enriched_query,
            param=QueryParam(
                mode="hybrid",
                only_need_context=True,
                top_k=top_k,
            ),
        )

        import re as _re_diag

        raw_context = result if isinstance(result, str) else str(result)
        raw_chunk_count = len(_re_diag.findall(r'\{"reference_id"', raw_context))
        print(f"DEBUG — LightRAG hybrid raw_chunks={raw_chunk_count}, raw_len={len(raw_context)}")

        # Naive fallback for step-specific queries when hybrid returns 0 text chunks.
        #
        # When hybrid returns 0 vector chunks it falls back to entity-related chunks,
        # which are biased toward Step-1 entities (ФИО/ИИН dominate the graph).
        # Naive mode skips graph traversal and uses direct vector similarity against
        # the indexed text chunks, giving step-specific content a fair ranking.
        # We use a minimal natural-language query so the embedding closely matches
        # the document step content rather than the metadata-tag-heavy enriched query.
        if raw_chunk_count == 0 and current_step is not None:
            naive_query = f"шаг {current_step} {query}"
            naive_result = await rag.aquery(
                naive_query,
                param=QueryParam(mode="naive", only_need_context=True, top_k=top_k),
            )
            naive_raw = naive_result if isinstance(naive_result, str) else str(naive_result)
            naive_chunks = len(_re_diag.findall(r'\{"reference_id"', naive_raw))
            print(f"DEBUG — naive fallback step={current_step}: found {naive_chunks} chunks")
            if naive_chunks > 0:
                raw_context = naive_raw
                raw_chunk_count = naive_chunks

        # Filter out chunks from unrelated services to avoid mixed templates
        context = raw_context
        if intent:
            context = _filter_context_by_intent(raw_context, intent, current_step=current_step)

        log_rag_search(query, intent or "none", language or "auto", len(context.split("\n")))

        filtered_chunk_count = len(_re_diag.findall(r'\{"reference_id"', context))
        print(f"DEBUG — after_filter chunks={filtered_chunk_count}, context_len={len(context)}, preview={context[:200] if context else 'EMPTY'}")

        if intent and raw_chunk_count > 0 and filtered_chunk_count == 0:
            print(f"⚠️  All {raw_chunk_count} chunks filtered out for intent={intent}. Check intent tags in indexed documents.")
        elif raw_chunk_count == 0:
            print(f"⚠️  Both hybrid and naive returned 0 text chunks. top_k={top_k}, intent={intent}, step={current_step}")

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
