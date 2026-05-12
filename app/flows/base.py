from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.schemas import FlowState


@dataclass
class FlowContext:
    message: str
    language: str
    history: list[dict]
    page: Optional[str]
    state: "FlowState"


class BaseFlow:
    flow_type: str = "base"
    requires_entity: bool = False
    max_steps: int = 0

    def next_state(self, ctx: FlowContext) -> "FlowState":
        return ctx.state.model_copy()

    def build_query(self, ctx: FlowContext) -> str:
        return ctx.message

    def is_step_progression(self, message: str) -> bool:
        return False
