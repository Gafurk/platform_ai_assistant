from __future__ import annotations

import time
from typing import Optional

from app.models.schemas import FlowState

_DEFAULT_TIMEOUT = 86400  # 24 hours


class SessionManager:
    """In-memory session store.

    Holds per-session message history (capped at 20 entries / 10 turns)
    and flow state separately, with three update modes:
      - update()              → state + history (normal LLM turn)
      - update_state_only()   → state only (clarification gate, no bot message)
      - update_history_only() → history only (FAQ interruption preserves flow)
    """

    def __init__(self, session_timeout: int = _DEFAULT_TIMEOUT) -> None:
        self._history: dict[str, list[dict]] = {}
        self._states: dict[str, FlowState] = {}
        self._accessed: dict[str, float] = {}
        self._timeout = session_timeout

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> list[dict]:
        return list(self._history.get(session_id, []))

    def get_state(self, session_id: str) -> FlowState:
        if session_id not in self._states:
            self._states[session_id] = FlowState()
        return self._states[session_id]

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def update(
        self,
        session_id: str,
        state: FlowState,
        user_msg: str,
        bot_msg: str,
    ) -> None:
        self._touch(session_id)
        self._states[session_id] = state
        self._append(session_id, user_msg, bot_msg)

    def update_state_only(self, session_id: str, state: FlowState) -> None:
        """Persist state without recording a history entry (clarification Q)."""
        self._touch(session_id)
        self._states[session_id] = state

    def update_history_only(
        self, session_id: str, user_msg: str, bot_msg: str
    ) -> None:
        """Record history without changing flow state (FAQ interruption)."""
        self._touch(session_id)
        self._append(session_id, user_msg, bot_msg)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        now = time.time()
        expired = [
            sid for sid, t in self._accessed.items()
            if now - t > self._timeout
        ]
        for sid in expired:
            self._history.pop(sid, None)
            self._states.pop(sid, None)
            self._accessed.pop(sid, None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _touch(self, session_id: str) -> None:
        self._accessed[session_id] = time.time()

    def _append(self, session_id: str, user_msg: str, bot_msg: str) -> None:
        history = self._history.setdefault(session_id, [])
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": bot_msg})
        # FIX #4A: Reduce history from 20 to 10 entries (max 5 turns)
        # This prevents LLM from following historical patterns when processing long sessions
        self._history[session_id] = history[-10:]


# Module-level singleton — persists across FastAPI requests
sessions = SessionManager()
