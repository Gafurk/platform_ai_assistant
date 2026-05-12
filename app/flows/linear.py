from __future__ import annotations

from app.flows.base import BaseFlow, FlowContext
from app.models.schemas import FlowState

MAX_STEPS = 5

_NEXT_STEP_PHRASES = frozenset({
    "дальше", "далее", "следующий", "следующий шаг",
    "продолжай", "продолжи", "давай", "да",
    "келесі", "жалғастыр", "ары қарай",
})


class LinearFlow(BaseFlow):
    """Step-by-step linear flow (e.g. TU application, 5 steps)."""

    flow_type = "linear"
    requires_entity = True
    max_steps = MAX_STEPS

    def is_step_progression(self, message: str) -> bool:
        return message.strip().lower() in _NEXT_STEP_PHRASES

    def next_state(self, ctx: FlowContext) -> FlowState:
        state = ctx.state.model_copy()

        # Step has not started yet but entity is known → start at step 1
        if state.step is None and state.entity is not None:
            state.step = 1
            return state

        # Explicit "next" request → advance, capped at MAX_STEPS
        if self.is_step_progression(ctx.message) and state.step is not None:
            state.step = min(state.step + 1, MAX_STEPS)

        return state

    def build_query(self, ctx: FlowContext) -> str:
        entity = ctx.state.entity or ""
        step = ctx.state.step
        base = ctx.message
        if step:
            return f"шаг {step} {entity} {base}".strip()
        return f"{entity} {base}".strip()
