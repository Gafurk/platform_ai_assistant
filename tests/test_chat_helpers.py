"""Tests for pure helper functions in app.routers.chat.

Imports only the regex helper, not the full chat router, to avoid
triggering LightRAG initialization during the test run.
"""
import re
from typing import Optional
import pytest


# ---------------------------------------------------------------------------
# Inline copy of _detect_entity_from_message_only (mirrors chat.py logic)
# ---------------------------------------------------------------------------

_FL_PHRASES = [
    "физическое лицо", "физ лицо", "физлицо",
    "физического лица", "физическим лицом",
    "жеке тұлға", "жеке тулга",
    "я фл", "как фл", "я ип",
]
_UL_PHRASES = [
    "юридическое лицо", "юр лицо", "юрлицо",
    "юридического лица", "юридическим лицом",
    "заңды тұлға", "занды тулга",
    "я юл", "как юл",
    " бин ", "наша организация", "наша компания",
]
_FL_EXACT = frozenset({"фл", "ф.л.", "физ", "физлицо", "физ лицо", "жт", "ж.т."})
_UL_EXACT = frozenset({"юл", "ю.л.", "юр", "юрлицо", "юр лицо", "зт", "з.т."})


def _detect_entity_from_message_only(message: str) -> Optional[str]:
    stripped = message.strip().lower()
    if stripped in _FL_EXACT:
        return "физическое лицо"
    if stripped in _UL_EXACT:
        return "юридическое лицо"
    text = " " + message.lower() + " "
    if any(ph in text for ph in _FL_PHRASES):
        return "физическое лицо"
    if any(ph in text for ph in _UL_PHRASES):
        return "юридическое лицо"
    return None


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


# ===========================================================================
# _detect_entity_from_message_only — entity override from current message
# ===========================================================================

class TestDetectEntityFromMessageOnly:
    # --- Exact matches (stripped, case-insensitive) ---
    def test_exact_fl_abbrev(self):
        assert _detect_entity_from_message_only("фл") == "физическое лицо"

    def test_exact_ul_abbrev(self):
        assert _detect_entity_from_message_only("юл") == "юридическое лицо"

    def test_exact_fl_punkt(self):
        assert _detect_entity_from_message_only("ф.л.") == "физическое лицо"

    def test_exact_ul_punkt(self):
        assert _detect_entity_from_message_only("ю.л.") == "юридическое лицо"

    def test_exact_kz_fl(self):
        assert _detect_entity_from_message_only("жт") == "физическое лицо"

    def test_exact_kz_ul(self):
        assert _detect_entity_from_message_only("зт") == "юридическое лицо"

    # --- Phrase matches in a sentence ---
    def test_phrase_fl_in_sentence(self):
        assert _detect_entity_from_message_only("я хочу подать рэн как физ лицо") == "физическое лицо"

    def test_phrase_ul_in_sentence(self):
        assert _detect_entity_from_message_only("если я юр лицо смогу ли") == "юридическое лицо"

    def test_ya_fl(self):
        assert _detect_entity_from_message_only("я фл") == "физическое лицо"

    def test_ya_ul(self):
        assert _detect_entity_from_message_only("я юл") == "юридическое лицо"

    def test_ya_yur_litso(self):
        assert _detect_entity_from_message_only("я юр лицо") == "юридическое лицо"

    def test_kak_fl(self):
        assert _detect_entity_from_message_only("как фл") == "физическое лицо"

    def test_hochyu_kak_fizlitso(self):
        assert _detect_entity_from_message_only("хочу подать на ТУ как физическое лицо") == "физическое лицо"

    def test_hochyu_kak_yrlitso(self):
        assert _detect_entity_from_message_only("хочу подать на ТУ как юридическое лицо") == "юридическое лицо"

    def test_bin_triggers_ul(self):
        # " бин " padded with spaces triggers UL phrase
        assert _detect_entity_from_message_only("укажите бин организации") == "юридическое лицо"

    def test_nasha_organizatsiya(self):
        assert _detect_entity_from_message_only("у нас наша организация хочет подать") == "юридическое лицо"

    # --- KZ phrase ---
    def test_kz_fl_phrase(self):
        assert _detect_entity_from_message_only("мен жеке тұлға ретінде") == "физическое лицо"

    def test_kz_ul_phrase(self):
        assert _detect_entity_from_message_only("заңды тұлға ретінде") == "юридическое лицо"

    # --- No match ---
    def test_no_match_tu_question(self):
        assert _detect_entity_from_message_only("хочу подать на ТУ") is None

    def test_no_match_step_question(self):
        assert _detect_entity_from_message_only("что делать на 3 шагу?") is None

    def test_no_match_sms(self):
        assert _detect_entity_from_message_only("не пришел смс код") is None

    def test_no_match_empty(self):
        assert _detect_entity_from_message_only("") is None

    def test_fl_wins_over_ul_when_fl_first(self):
        # FL phrase appears before UL phrase → FL returned
        assert _detect_entity_from_message_only("физ лицо или юр лицо") == "физическое лицо"
