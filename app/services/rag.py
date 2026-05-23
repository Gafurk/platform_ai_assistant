from app.services import lightrag_service


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
    Search the knowledge graph for context using LightRAG.

    This function delegates to lightrag_service.search_docs() which uses
    LightRAG's graph-based retrieval with only_need_context=True.

    Args:
        query: User question
        top_k: Number of nodes to retrieve
        intent: "tu_application" or "real_estate"
        entity: "физическое лицо" or "юридическое лицо"
        page: Page context (optional)
        language: "ru" or "kz"
        current_step: Current step number for multi-step processes

    Returns:
        Context string formatted for the LLM
    """
    return await lightrag_service.search_docs(
        query=query,
        top_k=top_k,
        intent=intent,
        entity=entity,
        page=page,
        language=language,
        current_step=current_step,
    )