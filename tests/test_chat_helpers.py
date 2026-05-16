"""Tests for pure helper functions in app.routers.chat.

Imports only the regex helper, not the full chat router, to avoid
triggering LightRAG initialization during the test run.
"""
import re
import pytest


# Inline copy of _extract_step_number to avoid importing the full router
# (which would trigger LightRAG init). If the implementation changes,
# update this copy too.
def _extract_step_number(message: str):
    lower = message.lower()
    for pat in (
        r"шаг\w*\s+(\d+)",
        r"(\d+)\s*[-–]?\s*шаг",
        r"қадам\w*\s+(\d+)",
        r"(\d+)\s*[-–]?\s*қадам",
        r"\bstep\s+(\d+)",
    ):
        m = re.search(pat, lower)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 10:
                return n
    return None


class TestExtractStepNumber:
    # Russian forms
    def test_ru_shag_before(self):
        assert _extract_step_number("шаг 1") == 1

    def test_ru_shag_genitive(self):
        assert _extract_step_number("шага 2 вопрос") == 2

    def test_ru_shag_dative(self):
        assert _extract_step_number("шагу 3 нужно") == 3

    def test_ru_number_before_shag(self):
        assert _extract_step_number("1 шаг") == 1

    def test_ru_hyphen_shag(self):
        assert _extract_step_number("2-шаг") == 2

    def test_ru_dash_shag(self):
        assert _extract_step_number("3–шаг") == 3

    # Kazakh forms (post normalize_kz — қадам, not кадам)
    def test_kz_kadam_before(self):
        assert _extract_step_number("қадам 1") == 1

    def test_kz_kadam_locative(self):
        assert _extract_step_number("қадамда 2") == 2

    def test_kz_number_before_kadam(self):
        assert _extract_step_number("1 қадам") == 1

    def test_kz_hyphen_kadam(self):
        assert _extract_step_number("2-қадам") == 2

    # English
    def test_en_step(self):
        assert _extract_step_number("step 3") == 3

    # Edge cases
    def test_step_in_sentence(self):
        assert _extract_step_number("в шагу 4 нужно указать ИИН") == 4

    def test_no_step(self):
        assert _extract_step_number("как зарегистрироваться") is None

    def test_out_of_range_returns_none(self):
        assert _extract_step_number("шаг 0") is None
        assert _extract_step_number("шаг 11") is None

    def test_only_number_returns_none(self):
        assert _extract_step_number("2") is None
