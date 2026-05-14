from __future__ import annotations

import logging

from app.flows.base import BaseFlow, FlowContext
from app.models.schemas import FlowState

# To see these debug logs set the root logger level to DEBUG in app/utils/logger.py
_log = logging.getLogger(__name__)

# Maps explicit user choice phrases to a human-readable situation label.
# The label is stored in FlowState.situation and passed to the LLM prompt.
_SITUATION_KEYWORDS: dict[str, list[str]] = {
    "через кадастровый номер": [
        "кадастр арқылы", "кадастр аркылы", "через кадастр",
        "по кадастру", "кадастрмен", "кадастровым номером",
    ],
    "через адресный регистр": [
        "адресный регистр", "адрестік регистр",
        "через адрес", "адрес арқылы", "адрес аркылы",
        "мекенжай арқылы", "мекенжай аркылы", "адресный",
    ],
}

# Question words that indicate the user is asking about something,
# not making a situation choice — skip situation detection for these.
_QUESTION_WORDS = frozenset({
    "как", "где", "откуда", "что такое",
    "қалай", "калай", "қайда", "кайда", "қайдан", "кайдан",
    "как получить", "как узнать", "как найти",
    "қалай алуға", "калай алуга", "қалай білуге",
})


class ScenarioFlow(BaseFlow):
    """Branching scenario flow (e.g. Real Estate — multiple situations)."""

    flow_type = "scenario"
    requires_entity = False  # Real estate process is identical for ФЛ/ЮЛ

    def next_state(self, ctx: FlowContext) -> FlowState:
        _log.debug("ScenarioFlow.next_state | msg=%r | situation=%r", ctx.message, ctx.state.situation)

        # Once situation is chosen, preserve it — never overwrite.
        if ctx.state.situation is not None:
            _log.debug("  → situation already set, skipping detection")
            return ctx.state.model_copy()

        msg_lower = ctx.message.lower()

        # Questions are not situation choices ("как получить кадастровый номер"
        # should not be treated as choosing the cadastral path).
        matched_qw = next((qw for qw in _QUESTION_WORDS if qw in msg_lower), None)
        if matched_qw:
            _log.debug("  → question-word guard fired on %r, no situation set", matched_qw)
            return ctx.state.model_copy()

        for situation_label, keywords in _SITUATION_KEYWORDS.items():
            matched_kw = next((kw for kw in keywords if kw in msg_lower), None)
            if matched_kw:
                _log.debug("  → matched keyword %r → situation=%r", matched_kw, situation_label)
                oq = f"{ctx.state.original_question or ''} {ctx.message}".strip()
                return ctx.state.model_copy(update={
                    "situation": situation_label,
                    "original_question": oq,
                })

        _log.debug("  → no keyword matched, situation remains None")
        return ctx.state.model_copy()

    def build_query(self, ctx: FlowContext) -> str:
        entity = ctx.state.entity or ""
        # Use the saved original question as the semantic anchor for RAG retrieval.
        # After situation is detected, original_question includes the user's
        # situational answer (e.g. "кадастр"), so RAG finds the right chunks.
        base = ctx.state.original_question or ctx.message
        return f"{entity} {base}".strip()
