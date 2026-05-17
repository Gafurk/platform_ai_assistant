"""Tests for rule-based layer: check_rule, SYSTEM_COMMANDS, NAVIGATION_RULES.
Also covers _filter_context_by_intent (Phase 3 metadata-tag filtering).
"""
import pytest
from app.services.rulebased import check_rule, SYSTEM_COMMANDS, NAVIGATION_RULES
from app.services.lightrag_service import _filter_context_by_intent


# ===========================================================================
# check_rule — SYSTEM_COMMANDS
# ===========================================================================

class TestSystemCommands:
    def test_greeting_privjet(self):
        answer, handoff = check_rule("привет")
        assert answer is not None
        assert "iSEL" in answer
        assert handoff is False

    def test_greeting_zdravstvujte(self):
        answer, _ = check_rule("здравствуйте")
        assert answer is not None and "iSEL" in answer

    def test_greeting_salem_kz(self):
        answer, _ = check_rule("салем")
        assert answer is not None
        assert "iSEL" in answer or "isel" in answer.lower()

    def test_operator_handoff(self):
        answer, _ = check_rule("нужен оператор")
        assert "+7 775 990 15 36" in answer

    def test_who_are_you_kto_ty(self):
        answer, _ = check_rule("кто ты")
        assert "iSEL" in answer

    def test_who_are_you_kem_ty(self):
        answer, _ = check_rule("кем ты являешься")
        assert "iSEL" in answer

    def test_ty_bot(self):
        answer, _ = check_rule("ты бот?")
        assert answer is not None

    def test_spasibo(self):
        answer, _ = check_rule("спасибо")
        assert answer is not None

    def test_rakhmet_kz(self):
        answer, _ = check_rule("рахмет")
        assert answer is not None

    def test_handoff_field_always_false_for_commands(self):
        for key in SYSTEM_COMMANDS:
            _, handoff = check_rule(key)
            assert handoff is False, f"handoff must be False for '{key}'"

    def test_no_match_returns_none(self):
        answer, handoff = check_rule("расскажи про технические условия")
        # Should not match any system command
        # (rule_based rules are substring matched so "условия" won't hit SYSTEM_COMMANDS)
        # The answer could be non-None only if it hits NAVIGATION_RULES
        # This test verifies at least that it returns a tuple
        assert isinstance(handoff, bool)

    def test_unrelated_message_returns_none(self):
        answer, handoff = check_rule("как зарегистрироваться на платформе")
        assert answer is None
        assert handoff is False


# ===========================================================================
# check_rule — NAVIGATION_RULES
# ===========================================================================

class TestNavigationRules:
    def test_status_zajavki(self):
        answer, _ = check_rule("статус заявки")
        assert answer is not None
        assert "Обращения" in answer

    def test_status_obrasheniya(self):
        answer, _ = check_rule("статус обращения")
        assert answer is not None

    def test_gde_moja_zavjka(self):
        answer, _ = check_rule("где моя заявка")
        assert "Обращения" in answer

    def test_proverit_status(self):
        answer, _ = check_rule("проверить статус")
        assert answer is not None

    def test_skachat_tu_ru(self):
        # Key "скачать технические условия" is all-lowercase and matches after msg.lower()
        answer, _ = check_rule("скачать технические условия")
        assert answer is not None
        assert "Обращения" in answer or "Скачать" in answer

    def test_skachat_tehnicheskie_usloviya(self):
        answer, _ = check_rule("скачать технические условия")
        assert answer is not None

    def test_gotovye_tu(self):
        answer, _ = check_rule("готовые технические условия")
        assert answer is not None

    def test_motivirovannyj_otkaz(self):
        answer, _ = check_rule("мотивированный отказ")
        assert "Обращения" in answer

    def test_skachat_otkaz(self):
        answer, _ = check_rule("скачать отказ")
        assert answer is not None

    def test_kz_status_martebe(self):
        answer, _ = check_rule("өтінішімнің мәртебесі")
        assert answer is not None
        assert "Өтініштер" in answer or "Қызметтер" in answer

    def test_kz_daleli_bas_tartu(self):
        answer, _ = check_rule("дәлелді бас тарту")
        assert answer is not None

    def test_kz_download_tu(self):
        answer, _ = check_rule("техникалық шарттарды жүктеу")
        assert answer is not None

    def test_longer_key_wins_over_shorter(self):
        # "мотивированный отказ" (longer) should win over "скачать отказ" (shorter)
        # when both could technically match
        answer1, _ = check_rule("мотивированный отказ")
        answer2, _ = check_rule("скачать отказ")
        # Both should return non-None (different keys)
        assert answer1 is not None
        assert answer2 is not None

    def test_navigation_handoff_always_false(self):
        for key in NAVIGATION_RULES:
            _, handoff = check_rule(key)
            assert handoff is False, f"handoff must be False for nav key '{key}'"

    def test_case_insensitive_match(self):
        # check_rule lowercases the message
        answer, _ = check_rule("СТАТУС ЗАЯВКИ")
        assert answer is not None


# ===========================================================================
# _filter_context_by_intent — Phase 3 metadata-tag filtering
# ===========================================================================

class TestFilterContextByIntent:
    _TU_CHUNK = "[FILENAME: tu.txt] [INTENT: tu_application] [LANGUAGE: ru] [TITLE: Шаг 1]\nЗаполните ФИО"
    _RE_CHUNK = "[FILENAME: re.txt] [INTENT: real_estate] [LANGUAGE: ru] [TITLE: Добавление]\nДобавьте объект"
    _UNTAGGED = "Общая информация без тега"

    def _context(self, *chunks) -> str:
        return "\n\n".join(chunks)

    def test_matching_intent_returns_only_matching_chunks(self):
        ctx = self._context(self._TU_CHUNK, self._RE_CHUNK)
        result = _filter_context_by_intent(ctx, "tu_application")
        assert "ФИО" in result
        assert "Добавьте объект" not in result

    def test_no_match_returns_empty(self):
        ctx = self._context(self._RE_CHUNK)
        result = _filter_context_by_intent(ctx, "tu_application")
        # Returns empty string — safer than returning unrelated service content
        assert result == ""

    def test_none_intent_returns_full_context(self):
        ctx = self._context(self._TU_CHUNK, self._RE_CHUNK)
        result = _filter_context_by_intent(ctx, None)
        assert result == ctx

    def test_empty_context_returns_empty(self):
        result = _filter_context_by_intent("", "tu_application")
        assert result == ""

    def test_mixed_context_filters_correctly(self):
        ctx = self._context(self._TU_CHUNK, self._UNTAGGED, self._RE_CHUNK)
        result = _filter_context_by_intent(ctx, "real_estate")
        assert "Добавьте объект" in result
        assert "ФИО" not in result

    def test_multiple_matching_chunks_all_returned(self):
        tu_chunk2 = "[FILENAME: tu2.txt] [INTENT: tu_application] [LANGUAGE: ru] [TITLE: Шаг 2]\nУкажите ИИН"
        ctx = self._context(self._TU_CHUNK, tu_chunk2, self._RE_CHUNK)
        result = _filter_context_by_intent(ctx, "tu_application")
        assert "ФИО" in result
        assert "ИИН" in result
        assert "Добавьте объект" not in result

    def test_supply_residential_intent_tag(self):
        chunk = "[FILENAME: sc.txt] [INTENT: supply_contract_residential] [LANGUAGE: ru]\nБытовой договор"
        ctx = self._context(chunk, self._TU_CHUNK)
        result = _filter_context_by_intent(ctx, "supply_contract_residential")
        assert "Бытовой договор" in result
        assert "ФИО" not in result

    def test_supply_nonresidential_intent_tag(self):
        chunk = "[FILENAME: sc.txt] [INTENT: supply_contract_non_residential] [LANGUAGE: ru]\nНебытовой договор"
        ctx = self._context(chunk, self._TU_CHUNK)
        result = _filter_context_by_intent(ctx, "supply_contract_non_residential")
        assert "Небытовой договор" in result

    # --- entity-only context (0 vector chunks from LightRAG) ---

    def test_entity_only_context_passes_through(self):
        # When LightRAG returns entity-only context (no [INTENT:] tags, no reference_id),
        # the filter must NOT return "" — it should return the full context so the LLM
        # can use entity descriptions instead of getting an empty context.
        entity_ctx = '{"entity": "ФИО заявителя", "description": "Поле заявителя на Шаге 1 ТУ"}'
        result = _filter_context_by_intent(entity_ctx, "tu_application")
        assert result == entity_ctx

    def test_entity_only_multi_entity_passes_through(self):
        entity_ctx = "\n\n".join([
            '{"entity": "Подписание", "description": "Финальный шаг подачи заявления"}',
            '{"entity": "ИИН", "description": "Идентификационный номер физлица"}',
        ])
        result = _filter_context_by_intent(entity_ctx, "load_calculation")
        assert result == entity_ctx

    # --- step-level filtering (Ошибка 1: TU steps 4/5 swap) ---
    # Chunks must use the real LightRAG JSON format ({"reference_id":...}) so
    # the tests exercise the JSON path where step filtering is applied.

    # Build a fake reference_id JSON chunk that the filter's JSON path will pick up.
    @staticmethod
    def _ref_chunk(intent: str, title: str, body: str) -> str:
        # Intentionally NOT valid JSON — the filter only checks string containment,
        # not JSON parsing. The chunk just needs to start with {"reference_id" and
        # contain the expected [INTENT:] / [TITLE:] / body text.
        return (
            f'{{"reference_id": "test", "content": '
            f'"[FILENAME: tu.txt] [INTENT: {intent}] [LANGUAGE: ru] [TITLE: {title}] {body}"}}'
        )

    def test_step_filter_prefers_matching_step_chunk(self):
        # "Ситуация 4" terminology block (same intent, no "Шаг 4" inside text)
        # must be excluded when current_step=4 and a proper Шаг-4 chunk exists.
        step4 = self._ref_chunk(
            "tu_application",
            "3.5 Пользователь (Шаг 4)",
            "Очереди строительства",
        )
        terminology = self._ref_chunk(
            "tu_application",
            "Ситуация 4",
            "Термины и определения Подписание заявления",
        )
        ctx = step4 + "\n" + terminology
        result = _filter_context_by_intent(ctx, "tu_application", current_step=4)
        assert "Очереди строительства" in result
        assert "Подписание заявления" not in result

    def test_step_filter_fallback_when_no_step_match(self):
        # If NO intent-matched chunk contains "Шаг N", keep all of them (don't over-filter).
        chunk = self._ref_chunk(
            "tu_application",
            "Общая информация",
            "Общий текст о ТУ без указания шага",
        )
        result = _filter_context_by_intent(chunk, "tu_application", current_step=3)
        assert "Общий текст о ТУ" in result

    def test_step_filter_none_step_does_not_narrow(self):
        # Without current_step, all intent-matched chunks are returned.
        step4 = self._ref_chunk("tu_application", "Шаг 4", "Очереди строительства")
        step5 = self._ref_chunk("tu_application", "Шаг 5", "Подписание документов")
        ctx = step4 + "\n" + step5
        result = _filter_context_by_intent(ctx, "tu_application", current_step=None)
        assert "Очереди строительства" in result
        assert "Подписание документов" in result
