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
        if ctx.history:
            user_msgs = [m["content"] for m in ctx.history if m["role"] == "user"]
            if user_msgs:
                best = max(user_msgs, key=len)
                # Use the richest historical message as semantic anchor only when the
                # current message is a short clarification reply (e.g. "ЖТ", "ЗТ").
                # This prevents the original intent question from being lost.
                if len(best) > len(ctx.message) * 2:
                    return f"{entity} {best} {ctx.message}".strip()
        return f"{entity} {ctx.message}".strip()
