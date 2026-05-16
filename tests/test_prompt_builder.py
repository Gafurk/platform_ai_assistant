"""Tests for PromptBuilder — static prompts, navigation facts, dynamic assembly."""
import pytest
from app.services.prompt_builder import (
    _STATIC_RU,
    _STATIC_KZ,
    _build_nav_facts,
    build_system_prompt,
)
from app.config.service_registry import ServiceRegistry

_registry = ServiceRegistry()


# ===========================================================================
# Static prompt structure
# ===========================================================================

class TestStaticPromptStructure:
    def test_ru_prompt_has_role_tag(self):
        assert "<role>" in _STATIC_RU

    def test_kz_prompt_has_role_tag(self):
        assert "<role>" in _STATIC_KZ

    def test_ru_prompt_has_language_rule(self):
        assert "<language_rule>" in _STATIC_RU

    def test_kz_prompt_has_language_rule(self):
        assert "<language_rule>" in _STATIC_KZ

    def test_ru_prompt_has_intent_handling(self):
        assert "<intent_handling>" in _STATIC_RU

    def test_kz_prompt_has_intent_handling(self):
        assert "<intent_handling>" in _STATIC_KZ

    def test_ru_prompt_has_navigation_facts(self):
        assert "<navigation_facts>" in _STATIC_RU

    def test_kz_prompt_has_navigation_facts(self):
        assert "<navigation_facts>" in _STATIC_KZ

    def test_ru_prompt_has_rules(self):
        assert "<rules>" in _STATIC_RU

    def test_kz_prompt_has_rules(self):
        assert "<rules>" in _STATIC_KZ

    def test_ru_prompt_has_formatting(self):
        assert "<formatting>" in _STATIC_RU

    def test_kz_prompt_has_formatting(self):
        assert "<formatting>" in _STATIC_KZ

    def test_ru_and_kz_prompts_are_different(self):
        assert _STATIC_RU != _STATIC_KZ

    def test_ru_prompt_mentions_russian_language(self):
        assert "РУССКИЙ" in _STATIC_RU or "русский" in _STATIC_RU.lower()

    def test_kz_prompt_mentions_kazakh_language(self):
        assert "КАЗАХСКИЙ" in _STATIC_KZ or "казахский" in _STATIC_KZ.lower()


# ===========================================================================
# Navigation facts — config-driven
# ===========================================================================

class TestNavigationFacts:
    def test_ru_nav_facts_contain_tu_path(self):
        nav = _build_nav_facts("ru")
        assert "Электроснабжение" in nav or "электроснабжен" in nav.lower()

    def test_kz_nav_facts_contain_tu_path(self):
        nav = _build_nav_facts("kz")
        assert "Электрмен жабдықтау" in nav or "электрмен" in nav.lower()

    def test_ru_nav_facts_contain_status_hint(self):
        nav = _build_nav_facts("ru")
        assert "Обращения" in nav or "статус" in nav.lower()

    def test_kz_nav_facts_contain_status_hint(self):
        nav = _build_nav_facts("kz")
        assert "Өтініштер" in nav or "мәртебе" in nav.lower()

    def test_nav_facts_include_all_services_with_paths(self):
        nav_ru = _build_nav_facts("ru")
        for service_id in _registry.list_services():
            svc = _registry.get_service(service_id)
            path = svc.navigation_path.get("ru") or svc.navigation_path.get("kz") or ""
            if path:
                # The path should appear somewhere in nav facts
                assert any(segment in nav_ru for segment in path.split("→")[:1])


# ===========================================================================
# Intent handling blocks
# ===========================================================================

class TestIntentHandlingBlocks:
    def test_ru_prompt_contains_tu_block(self):
        assert "технические условия" in _STATIC_RU.lower() or "ТУ" in _STATIC_RU

    def test_kz_prompt_contains_tu_block(self):
        assert "техникалық шарт" in _STATIC_KZ.lower() or "ТШ" in _STATIC_KZ

    def test_ru_prompt_contains_real_estate_block(self):
        assert "недвижимости" in _STATIC_RU.lower()

    def test_kz_prompt_contains_real_estate_block(self):
        assert "жылжымайтын" in _STATIC_KZ.lower()

    def test_ru_prompt_contains_supply_residential(self):
        assert "бытовой" in _STATIC_RU.lower()

    def test_ru_prompt_contains_supply_nonresidential(self):
        assert "небытовой" in _STATIC_RU.lower()

    def test_ru_prompt_contains_faq_fallback(self):
        assert "FAQ" in _STATIC_RU or "не определён" in _STATIC_RU or "интент не определён" in _STATIC_RU

    def test_ru_rules_count_at_least_10(self):
        # Rules section has numbered items
        import re
        rules_start = _STATIC_RU.find("<rules>")
        rules_end = _STATIC_RU.find("</rules>")
        rules_text = _STATIC_RU[rules_start:rules_end]
        numbered = re.findall(r"^\d+\.", rules_text, re.MULTILINE)
        assert len(numbered) >= 10


# ===========================================================================
# build_system_prompt — dynamic assembly
# ===========================================================================

class TestBuildSystemPromptBase:
    def test_no_dynamic_parts_returns_static_ru(self):
        prompt = build_system_prompt("ru", "", None, None, None, None)
        assert prompt == _STATIC_RU

    def test_no_dynamic_parts_returns_static_kz(self):
        prompt = build_system_prompt("kz", "", None, None, None, None)
        assert prompt == _STATIC_KZ

    def test_unknown_language_falls_back_to_ru(self):
        prompt = build_system_prompt("en", "", None, None, None, None)
        assert prompt == _STATIC_RU


class TestBuildSystemPromptHistory:
    def test_history_tag_present_when_non_empty(self):
        prompt = build_system_prompt("ru", "Пользователь: привет\nАссистент: здравствуйте", None, None, None, None)
        assert "<history>" in prompt
        assert "привет" in prompt

    def test_no_history_tag_when_empty_string(self):
        prompt = build_system_prompt("ru", "", None, None, None, None)
        assert "<history>" not in prompt


class TestBuildSystemPromptIntentContext:
    def test_intent_context_tag_present_with_intent(self):
        prompt = build_system_prompt("ru", "", "tu_application", None, None, None)
        assert "<intent_context>" in prompt

    def test_intent_context_includes_entity(self):
        prompt = build_system_prompt("ru", "", "tu_application", "физическое лицо", None, None)
        assert "физическое лицо" in prompt

    def test_intent_context_includes_page(self):
        prompt = build_system_prompt("ru", "", None, None, None, "шаг 1")
        assert "<intent_context>" in prompt
        assert "шаг 1" in prompt

    def test_intent_context_includes_situation(self):
        prompt = build_system_prompt("ru", "", "real_estate", None, None, None, situation="через кадастровый номер")
        assert "через кадастровый номер" in prompt

    def test_no_intent_context_when_all_none(self):
        # When all dynamic params are None/empty, prompt equals the pre-built static
        prompt = build_system_prompt("ru", "", None, None, None, None)
        assert prompt == _STATIC_RU

    def test_tu_display_name_in_intent_context(self):
        prompt = build_system_prompt("ru", "", "tu_application", None, None, None)
        # The RU display name for tu_application should appear in intent context
        svc = _registry.get_service("tu_application")
        name = svc.display_name.get("ru", "")
        assert name in prompt or "технические условия" in prompt.lower()


class TestBuildSystemPromptStepControl:
    def test_step_control_tag_present_with_step(self):
        prompt = build_system_prompt("ru", "", "tu_application", "физическое лицо", 2, None)
        assert "<step_control>" in prompt

    def test_step_control_shows_current_step(self):
        prompt = build_system_prompt("ru", "", "tu_application", "физическое лицо", 3, None)
        assert "3" in prompt

    def test_step_control_shows_max_steps(self):
        svc = _registry.get_service("tu_application")
        max_steps = svc.flow.max_steps
        prompt = build_system_prompt("ru", "", "tu_application", "физическое лицо", 2, None)
        assert str(max_steps) in prompt

    def test_intermediate_step_has_next_button_hint(self):
        prompt = build_system_prompt("ru", "", "tu_application", "физическое лицо", 2, None)
        assert "Далее" in prompt or "следующему шагу" in prompt.lower()

    def test_final_step_has_last_step_message(self):
        svc = _registry.get_service("tu_application")
        max_steps = svc.flow.max_steps
        prompt = build_system_prompt("ru", "", "tu_application", "физическое лицо", max_steps, None)
        assert "ПОСЛЕДНИЙ шаг" in prompt or "последний" in prompt.lower()

    def test_final_step_mentions_obrasheniya(self):
        svc = _registry.get_service("tu_application")
        max_steps = svc.flow.max_steps
        prompt = build_system_prompt("ru", "", "tu_application", "физическое лицо", max_steps, None)
        assert "Обращения" in prompt

    def test_kz_intermediate_step_has_ari_qaray_hint(self):
        prompt = build_system_prompt("kz", "", "tu_application", "жеке тұлға", 1, None)
        assert "Әрі қарай" in prompt or "батырмасын" in prompt.lower()

    def test_no_step_control_when_step_none(self):
        prompt = build_system_prompt("ru", "", "tu_application", "физическое лицо", None, None)
        # The dynamic step_control block contains "Текущий шаг: N из M" — absent when step is None
        assert "Текущий шаг:" not in prompt

    def test_max_steps_from_config_not_hardcoded(self):
        # Verify the step ceiling comes from service config, not hardcoded 5
        svc = _registry.get_service("tu_application")
        max_steps = svc.flow.max_steps
        prompt_last = build_system_prompt("ru", "", "tu_application", "физическое лицо", max_steps, None)
        prompt_mid = build_system_prompt("ru", "", "tu_application", "физическое лицо", max_steps - 1, None)
        assert "ПОСЛЕДНИЙ шаг" in prompt_last
        assert "ПОСЛЕДНИЙ шаг" not in prompt_mid
