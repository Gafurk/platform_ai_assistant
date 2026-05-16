from __future__ import annotations

import re
from typing import Optional

from app.flows.base import BaseFlow, FlowContext
from app.models.schemas import FlowState

MAX_STEPS = 5

_NEXT_STEP_PHRASES = frozenset({
    "дальше", "далее", "следующий", "следующий шаг",
    "продолжай", "продолжи", "давай", "да",
    "келесі", "жалғастыр", "ары қарай",
})

# Matches messages that are ENTIRELY a step reference, e.g.:
#   "шаг 3", "3 шаг", "шагу 2", "3-шаг", "қадам 4", "4 кадамда", "3-қадам"
# Intentionally strict (requires full match) so "объясни снова шаг 2" is NOT caught.
_STEP_JUMP_RE = re.compile(
    r"^\s*(?:"
    r"(?P<n1>[1-5])\s*[-–]?\s*(?:шаг\w*|қадам\w*|кадам\w*)"
    r"|(?:шаг\w*|қадам\w*|кадам\w*)\s*[-–]?\s*(?P<n2>[1-5])"
    r")\s*$",
    re.IGNORECASE,
)


def _extract_step_jump(message: str) -> Optional[int]:
    """Return the step number if the entire message is a step reference, else None."""
    m = _STEP_JUMP_RE.match(message.strip())
    if not m:
        return None
    n = m.group("n1") or m.group("n2")
    return int(n) if n else None


class LinearFlow(BaseFlow):
    """Step-by-step linear flow (e.g. TU application, 5 steps)."""

    flow_type = "linear"
    requires_entity = True
    max_steps = MAX_STEPS

    def is_step_progression(self, message: str) -> bool:
        return (
            message.strip().lower() in _NEXT_STEP_PHRASES
            or _extract_step_jump(message) is not None
        )

    def next_state(self, ctx: FlowContext) -> FlowState:
        state = ctx.state.model_copy()

        # Step has not started yet but entity is known → start at specified or step 1
        if state.step is None and state.entity is not None:
            jump = _extract_step_jump(ctx.message)
            state.step = min(jump, MAX_STEPS) if jump else 1
            return state

        if state.step is not None:
            jump = _extract_step_jump(ctx.message)
            if jump is not None:
                # Only jump forward — never regress
                if jump > state.step:
                    state.step = min(jump, MAX_STEPS)
                return state

            # Explicit "next" request → advance, capped at MAX_STEPS
            if self.is_step_progression(ctx.message):
                state.step = min(state.step + 1, MAX_STEPS)

        return state

    def build_query(self, ctx: FlowContext) -> str:
        entity = ctx.state.entity or ""
        step = ctx.state.step
        base = ctx.message
        if step:
            return f"шаг {step} {entity} {base}".strip()
        return f"{entity} {base}".strip()
