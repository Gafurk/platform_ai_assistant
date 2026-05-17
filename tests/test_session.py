"""Tests for SessionManager."""
import pytest

from app.models.schemas import FlowState
from app.services.session import SessionManager


def make_sm():
    return SessionManager(session_timeout=86400)


# ===========================================================================
# Default state
# ===========================================================================

def test_empty_session_returns_default_state():
    sm = make_sm()
    state = sm.get_state("s1")
    assert isinstance(state, FlowState)
    assert state.intent is None
    assert state.entity is None
    assert state.step is None
    assert state.locked is False


def test_empty_session_returns_empty_history():
    sm = make_sm()
    assert sm.get_history("s1") == []


# ===========================================================================
# update() — state + history
# ===========================================================================

def test_update_persists_state():
    sm = make_sm()
    state = FlowState(intent="tu_application", entity="физическое лицо", step=2)
    sm.update("s1", state, "user msg", "bot msg")
    retrieved = sm.get_state("s1")
    assert retrieved.intent == "tu_application"
    assert retrieved.entity == "физическое лицо"
    assert retrieved.step == 2


def test_update_appends_history():
    sm = make_sm()
    sm.update("s1", FlowState(), "hello", "hi")
    sm.update("s1", FlowState(), "second", "reply")
    history = sm.get_history("s1")
    assert len(history) == 4
    assert history[0] == {"role": "user", "content": "hello"}
    assert history[1] == {"role": "assistant", "content": "hi"}
    assert history[2] == {"role": "user", "content": "second"}
    assert history[3] == {"role": "assistant", "content": "reply"}


def test_history_capped_at_10_entries():
    sm = make_sm()
    for i in range(10):  # 10 turns = 20 entries → capped at 10
        sm.update("s1", FlowState(), f"u{i}", f"b{i}")
    history = sm.get_history("s1")
    assert len(history) == 10
    # Most recent entries preserved
    assert history[-1]["content"] == "b9"
    assert history[-2]["content"] == "u9"


def test_get_history_returns_copy():
    sm = make_sm()
    sm.update("s1", FlowState(), "msg", "reply")
    h1 = sm.get_history("s1")
    h1.append({"role": "user", "content": "injected"})
    h2 = sm.get_history("s1")
    assert len(h2) == 2  # original not mutated


# ===========================================================================
# update_history_only() — FAQ interruption preserves flow state
# ===========================================================================

def test_update_history_only_preserves_state():
    sm = make_sm()
    state = FlowState(intent="tu_application", step=3, locked=True)
    sm.update("s1", state, "start", "started")
    sm.update_history_only("s1", "faq question", "faq answer")

    assert sm.get_state("s1").step == 3
    assert sm.get_state("s1").intent == "tu_application"
    assert sm.get_state("s1").locked is True


def test_update_history_only_records_exchange():
    sm = make_sm()
    sm.update("s1", FlowState(), "flow msg", "flow reply")
    sm.update_history_only("s1", "faq q", "faq a")
    history = sm.get_history("s1")
    assert history[-1] == {"role": "assistant", "content": "faq a"}
    assert history[-2] == {"role": "user", "content": "faq q"}


# ===========================================================================
# update_state_only() — clarification gate
# ===========================================================================

def test_update_state_only_doesnt_add_history():
    sm = make_sm()
    state = FlowState(intent="tu_application", entity=None)
    sm.update_state_only("s1", state)
    assert sm.get_history("s1") == []


def test_update_state_only_persists_state():
    sm = make_sm()
    state = FlowState(intent="real_estate", entity="юридическое лицо")
    sm.update_state_only("s1", state)
    assert sm.get_state("s1").intent == "real_estate"
    assert sm.get_state("s1").entity == "юридическое лицо"


# ===========================================================================
# Session isolation
# ===========================================================================

def test_sessions_are_isolated():
    sm = make_sm()
    state1 = FlowState(intent="tu_application", step=1)
    state2 = FlowState(intent="real_estate", step=None)
    sm.update("s1", state1, "m1", "r1")
    sm.update("s2", state2, "m2", "r2")
    assert sm.get_state("s1").intent == "tu_application"
    assert sm.get_state("s2").intent == "real_estate"
    assert sm.get_history("s1") != sm.get_history("s2")


# ===========================================================================
# Cleanup
# ===========================================================================

def test_cleanup_removes_expired_sessions():
    sm = SessionManager(session_timeout=0)  # expire immediately
    sm.update("s_old", FlowState(), "msg", "reply")
    sm.cleanup()
    # After cleanup, getting state returns fresh default
    fresh = sm.get_state("s_old")
    assert fresh.intent is None
    assert fresh.step is None


def test_cleanup_preserves_active_sessions():
    sm = SessionManager(session_timeout=86400)
    state = FlowState(intent="tu_application", step=2)
    sm.update("s_active", state, "msg", "reply")
    sm.cleanup()
    assert sm.get_state("s_active").step == 2
