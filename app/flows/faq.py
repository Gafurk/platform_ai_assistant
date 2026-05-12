from __future__ import annotations

from app.flows.base import BaseFlow, FlowContext


class FAQFlow(BaseFlow):
    """Unstructured FAQ flow — pure retrieval, no step/situation tracking."""

    flow_type = "faq"
    requires_entity = False

    def build_query(self, ctx: FlowContext) -> str:
        return ctx.message
