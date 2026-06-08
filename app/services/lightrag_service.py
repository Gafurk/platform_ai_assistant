import os
import numpy as np
from openai import AsyncOpenAI
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from app.utils.logger import log_rag_search, log_qdrant_error
from dotenv import load_dotenv

load_dotenv()

_LLM_MODEL = os.getenv("OPENAI_MODEL", "")
# Use a dedicated non-reasoning model for LightRAG entity/relation extraction.
# Reasoning models (gpt-5-mini, o-series) consume thousands of reasoning tokens on
# LightRAG's 14 KB system prompt, exhausting max_completion_tokens before any visible
# output is produced → 0 entities extracted.  gpt-4.1-nano is fast and follows the
# <|#|> / <|COMPLETE|> format reliably.
# Set LIGHTRAG_EXTRACT_MODEL in .env to override; falls back to OPENAI_MODEL.
_EXTRACT_MODEL = os.getenv("LIGHTRAG_EXTRACT_MODEL") or _LLM_MODEL

# Separate clients with different timeouts
_oai_embeddings = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""), timeout=300.0)
_oai_llm = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""), timeout=600.0)


async def gpt41_nano_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] = [],
    **kwargs,
) -> str:
    """LightRAG entity/relation extraction. Uses LIGHTRAG_EXTRACT_MODEL (non-reasoning)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    try:
        response = await _oai_llm.chat.completions.create(
            model=_EXTRACT_MODEL,
            messages=messages,
            temperature=kwargs.get("temperature", 0),
            max_completion_tokens=kwargs.get("max_completion_tokens", 8000),
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


def _create_rag() -> LightRAG:
    return LightRAG(
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


rag = _create_rag()


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
            else:
                # Intent-matched chunks exist but none belong to this step.
                # Return empty so the step fallback in search_docs can retrieve
                # step-specific content via a broader query instead of serving an
                # off-topic chunk (e.g. "Ситуация 4" for a Шаг 5 query, or
                # a Шаг 2 chunk for a Шаг 1 query in РЭН).
                return ""
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


async def reinitialize():
    """Wipe data/lightrag/ and create a fresh LightRAG instance. Called by /reindex."""
    global rag
    import shutil

    lightrag_dir = "./data/lightrag"

    # Close the current rag instance to release any open file handles before wiping.
    # On Windows, rmtree silently fails (ignore_errors=True) when files are held open
    # by a concurrent search, leaving the old kv_store_doc_status.json intact.
    # initialize_storages() would then load stale "failed" entries, causing LightRAG
    # to treat fresh inserts as duplicates and skip all entity extraction.
    try:
        await rag.finalize_storages()
    except Exception:
        pass

    try:
        shutil.rmtree(lightrag_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to wipe {lightrag_dir}: {e}") from e

    rag = _create_rag()
    await rag.initialize_storages()
    print("✅ LightRAG reinitialized (fresh)")


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

        # Apply intent + step filter immediately after hybrid retrieval.
        # Filtering first (before the fallback check) lets us measure how many
        # step-relevant chunks hybrid actually returned.
        context = raw_context
        if intent:
            context = _filter_context_by_intent(raw_context, intent, current_step=current_step)

        filtered_chunk_count = len(_re_diag.findall(r'\{"reference_id"', context))
        print(f"DEBUG — after_filter chunks={filtered_chunk_count}, context_len={len(context)}, preview={context[:200] if context else 'EMPTY'}")

        # Step-specific fallback: when the hybrid result yields 0 step-specific chunks,
        # retry with a clean natural-language query and doubled top_k.
        #
        # Threshold is == 0 (not <= 1) because:
        #   - The strict step filter (else: return "") already ensures that any chunk
        #     that passes the filter genuinely belongs to "Шаг N" — a wrong chunk can
        #     never survive (e.g. "Ситуация 4" for step 4 returns "" not 1 chunk).
        #   - Each ingested step section is exactly 1 chunk (all sections fit within
        #     LightRAG's chunk_token_size=1200). When 1 correct chunk is found, it is
        #     complete and a second query is wasteful.
        #   - <= 1 caused every step request to make 2 LightRAG calls.
        #
        # Why clean query? The "[TITLE: Шаг N] [INTENT:]" prefix causes LightRAG's
        # entity extractor to anchor on "Шаг 1" (most-connected entity) for all steps.
        # Entity is intentionally excluded here: shared steps (2-N) have no FL/UL-specific
        # text, so including "физическое лицо" in the query causes LightRAG to rank
        # entity-connected chunks from other services (TU_application dominates the graph)
        # above the correct step chunk. A bare "Шаг N" query lets the extractor find the
        # step entity → graph traversal returns all Шаг-N chunks → intent filter keeps only
        # the correct service's chunk.
        # Why top_k * 2? The correct step chunk may rank 21+ and be cut off at top_k=20.
        if current_step is not None and filtered_chunk_count == 0:
            fallback_query = f"Шаг {current_step}"
            fallback_result = await rag.aquery(
                fallback_query,
                param=QueryParam(mode="hybrid", only_need_context=True, top_k=top_k * 2),
            )
            fallback_raw = fallback_result if isinstance(fallback_result, str) else str(fallback_result)
            fallback_context = (
                _filter_context_by_intent(fallback_raw, intent, current_step=current_step)
                if intent else fallback_raw
            )
            fallback_chunks = len(_re_diag.findall(r'\{"reference_id"', fallback_context))
            print(f"DEBUG — step fallback step={current_step} top_k={top_k * 2}: found {fallback_chunks} chunks (main had {filtered_chunk_count})")
            if fallback_chunks > filtered_chunk_count:
                context = fallback_context
                filtered_chunk_count = fallback_chunks

        log_rag_search(query, intent or "none", language or "auto", len(context.split("\n")))

        print(f"DEBUG — final chunks={filtered_chunk_count}, context_len={len(context)}")

        if intent and raw_chunk_count > 0 and filtered_chunk_count == 0:
            print(f"⚠️  All {raw_chunk_count} hybrid chunks filtered out for intent={intent}, step={current_step}. Fallback also found 0.")
        elif raw_chunk_count == 0:
            print(f"⚠️  LightRAG returned 0 entity-related chunks. top_k={top_k}, intent={intent}, step={current_step}")

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
