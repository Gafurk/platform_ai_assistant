from __future__ import annotations

from app.flows.base import BaseFlow, FlowContext
from app.models.schemas import FlowState


class ScenarioFlow(BaseFlow):
    """Branching scenario flow (e.g. Real Estate — multiple situations)."""

    flow_type = "scenario"
    requires_entity = True

    def next_state(self, ctx: FlowContext) -> FlowState:
        return ctx.state.model_copy()

    def build_query(self, ctx: FlowContext) -> str:
        entity = ctx.state.entity or ""
        # Include last user turn for context continuity — ensures clarification answers
        # land in the right retrieval space without dropping entity grounding.
        if ctx.history:
            last_user = next(
                (m["content"] for m in reversed(ctx.history) if m["role"] == "user"),
                "",
            )
            if last_user:
                return f"{entity} {last_user} {ctx.message}".strip()
        return f"{entity} {ctx.message}".strip()
