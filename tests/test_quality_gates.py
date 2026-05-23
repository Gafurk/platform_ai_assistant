"""
Unit tests for quality gates and LLM validation.

Tests the new quality_gates functionality in prompt_builder and validation in llm_service.
"""
import pytest
import asyncio
from app.services.prompt_builder import build_system_prompt, _QUALITY_GATES_RU, _QUALITY_GATES_KZ
from app.services.llm_service import _validate_question, _contains_keyword
from app.services.rulebased import SYSTEM_COMMANDS, NAVIGATION_RULES


class TestQualityGatesIntegration:
    """Test that quality gates are properly integrated into prompts."""

    def test_quality_gates_ru_present(self):
        """Test that Russian quality gates are in the prompt."""
        prompt = build_system_prompt(
            language="ru",
            history="",
            intent=None,
            entity=None,
            current_step=None,
            page=None,
            situation=None
        )
        assert "<quality_gates>" in prompt
        assert "OFF-TOPIC" in prompt
        assert "GOVERNMENT TERMS" in prompt

    def test_quality_gates_kz_present(self):
        """Test that Kazakh quality gates are in the prompt."""
        prompt = build_system_prompt(
            language="kz",
            history="",
            intent=None,
            entity=None,
            current_step=None,
            page=None,
            situation=None
        )
        assert "<quality_gates>" in prompt
        assert "OFF-TOPIC" in prompt
        assert "МЕМЛЕКЕТТІК ТЕРМИНДЕР" in prompt

    def test_rules_include_offtopic_ru(self):
        """Test that Russian rules include off-topic handling."""
        prompt = build_system_prompt(
            language="ru",
            history="",
            intent=None,
            entity=None,
            current_step=None,
            page=None,
            situation=None
        )
        assert "Я помогаю только с вопросами по платформе iSEL" in prompt

    def test_rules_include_offtopic_kz(self):
        """Test that Kazakh rules include off-topic handling."""
        prompt = build_system_prompt(
            language="kz",
            history="",
            intent=None,
            entity=None,
            current_step=None,
            page=None,
            situation=None
        )
        assert "iSEL платформасы" in prompt

    def test_no_predetermine_dialog_ru(self):
        """Test that prompt does not predetermine user intent (Russian)."""
        prompt = build_system_prompt(
            language="ru",
            history="",
            intent=None,
            entity=None,
            current_step=None,
            page=None,
            situation=None
        )
        # Should NOT tell LLM to assume user is new or start with step 1 on its own
        assert "Похоже, вы новый пользователь" not in prompt
        assert "Давайте начнём с шага 1" not in prompt


class TestValidationLogic:
    """Test the _validate_question pre-validation logic."""

    @pytest.mark.asyncio
    async def test_empty_question_ru(self):
        """Test that empty questions are rejected."""
        result = await _validate_question("   ", "", "ru")
        assert result is not None
        assert "задайте вопрос" in result[0].lower()

    @pytest.mark.asyncio
    async def test_empty_question_kz(self):
        """Test that empty questions are rejected in Kazakh."""
        result = await _validate_question("", "", "kz")
        assert result is not None

    @pytest.mark.asyncio
    async def test_offtopic_anekdot_ru(self):
        """Test that off-topic joke requests are caught."""
        result = await _validate_question("расскажи анекдот", "", "ru")
        assert result is not None
        assert result[1] is True  # is_off_topic flag

    @pytest.mark.asyncio
    async def test_offtopic_with_context_passes(self):
        """Test that off-topic keywords with context still pass (LLM will decide)."""
        # Even with off-topic keywords, if there's context from platform, let LLM decide
        result = await _validate_question("расскажи про процесс в смешном стиле", "section about TU application", "ru")
        # This should pass pre-validation because we have platform context
        # (soft heuristic, not hard reject)
        assert result is None  # None means validation passed

    @pytest.mark.asyncio
    async def test_valid_platform_question_ru(self):
        """Test that valid platform questions pass."""
        result = await _validate_question("как подать заявку на ТУ", "some context", "ru")
        assert result is None  # None means validation passed

    @pytest.mark.asyncio
    async def test_valid_platform_question_kz(self):
        """Test that valid Kazakh platform questions pass."""
        result = await _validate_question("өтініш қалай беру керек", "some context", "kz")
        assert result is None  # None means validation passed


class TestKeywordDetection:
    """Test keyword detection utilities."""

    def test_contains_keyword_basic(self):
        """Test basic keyword detection."""
        keywords = frozenset({"тест", "платформа", "заявка"})
        assert _contains_keyword("тестовая платформа", keywords) is True
        assert _contains_keyword("Платформа iSEL", keywords) is True
        assert _contains_keyword("кот собака", keywords) is False

    def test_contains_keyword_cyrillic(self):
        """Test keyword detection with Cyrillic."""
        keywords = frozenset({"ЭЦП", "электронно"})
        assert _contains_keyword("Подпишу документ ЭЦП", keywords) is True
        assert _contains_keyword("электронный подпис", keywords) is True


class TestRuleBasedKzSupport:
    """Test Kazakh support in rule-based layer."""

    def test_kazakh_offtopic_commands_exist(self):
        """Test that Kazakh off-topic commands are registered."""
        kz_commands = {
            "әнде": SYSTEM_COMMANDS.get("әнде"),
            "өлең": SYSTEM_COMMANDS.get("өлең"),
            "ән": SYSTEM_COMMANDS.get("ән"),
        }
        # At least some should exist
        existing = [k for k, v in kz_commands.items() if v is not None]
        assert len(existing) > 0, "No Kazakh off-topic commands found"

    def test_rule_structure_consistency(self):
        """Test that all rules have proper structure."""
        for key, val in SYSTEM_COMMANDS.items():
            assert isinstance(val, dict), f"Rule {key} is not a dict"
            assert "answer" in val, f"Rule {key} missing 'answer'"
            assert "handoff" in val, f"Rule {key} missing 'handoff'"
            assert isinstance(val["answer"], str), f"Rule {key} answer is not string"
            assert isinstance(val["handoff"], bool), f"Rule {key} handoff is not bool"


class TestGovernmentTermsHandling:
    """Test government terms (ЭЦП, eGov, etc.) handling."""

    def test_gov_terms_in_rules_ru(self):
        """Test that Russian rules mention government terms handling."""
        prompt = build_system_prompt(
            language="ru",
            history="",
            intent=None,
            entity=None,
            current_step=None,
            page=None,
            situation=None
        )
        assert "ЭЦП" in prompt
        assert "eGov" in prompt
        assert "БИН" in prompt

    def test_gov_terms_in_rules_kz(self):
        """Test that Kazakh rules mention government terms handling."""
        prompt = build_system_prompt(
            language="kz",
            history="",
            intent=None,
            entity=None,
            current_step=None,
            page=None,
            situation=None
        )
        assert "ЭЦҚ" in prompt or "eGov" in prompt

    def test_warning_marker_in_gov_terms(self):
        """Test that government terms have warning marker."""
        prompt = build_system_prompt(
            language="ru",
            history="",
            intent=None,
            entity=None,
            current_step=None,
            page=None,
            situation=None
        )
        # Should mention the warning marker for gov terms
        assert "⚠️" in prompt or "На основе общих знаний" in prompt


class TestBilingualConsistency:
    """Test that RU and KZ versions are consistent."""

    def test_both_languages_have_quality_gates(self):
        """Test that both languages have quality gates."""
        assert len(_QUALITY_GATES_RU) > 100, "Russian quality gates too short"
        assert len(_QUALITY_GATES_KZ) > 100, "Kazakh quality gates too short"

    def test_both_languages_have_offtopic_rules(self):
        """Test that both have off-topic handling in rules."""
        prompt_ru = build_system_prompt("ru", "", None, None, None, None)
        prompt_kz = build_system_prompt("kz", "", None, None, None, None)
        
        assert "OFF-TOPIC" in prompt_ru
        assert "OFF-TOPIC" in prompt_kz

    def test_no_duplicate_language_confusion(self):
        """Test that Russian prompt doesn't have Kazakh-only terms and vice versa."""
        prompt_ru = build_system_prompt("ru", "", None, None, None, None)
        prompt_kz = build_system_prompt("kz", "", None, None, None, None)
        
        # Russian should NOT have excessive Kazakh-only Cyrillic
        kz_chars = frozenset("әғқңөұүі")
        kz_char_count = sum(1 for c in prompt_ru if c.lower() in kz_chars)
        assert kz_char_count < 50, f"Russian prompt has too many Kazakh chars: {kz_char_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
