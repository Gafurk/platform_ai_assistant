"""Tests for flow engine: LinearFlow, ScenarioFlow, FAQFlow, registry."""
import pytest

from app.flows.base import FlowContext
from app.flows.faq import FAQFlow
from app.flows.linear import LinearFlow
from app.flows.registry import classify_intent, get_flow, has_flow_keywords
from app.flows.scenario import ScenarioFlow
from app.models.schemas import FlowState


def ctx(message, state=None, history=None, page=None, lang="ru"):
    return FlowContext(
        message=message,
        language=lang,
        history=history or [],
        page=page,
        state=state or FlowState(),
    )


# ===========================================================================
# LinearFlow — step progression
# ===========================================================================

class TestLinearFlowStepProgression:
    flow = LinearFlow()

    def test_step_starts_at_1_when_entity_set(self):
        state = FlowState(intent="tu_application", entity="физическое лицо", step=None)
        new = self.flow.next_state(ctx("подать заявку", state=state))
        assert new.step == 1

    def test_step_increments_on_progression_word(self):
        for word in ["дальше", "далее", "следующий", "да", "продолжай", "келесі"]:
            state = FlowState(intent="tu_application", entity="физическое лицо", step=2)
            new = self.flow.next_state(ctx(word, state=state))
            assert new.step == 3, f"step should advance to 3 for '{word}'"

    def test_step_capped_at_max(self):
        state = FlowState(intent="tu_application", entity="физическое лицо", step=5)
        new = self.flow.next_state(ctx("дальше", state=state))
        assert new.step == 5

    def test_step_not_started_without_entity(self):
        state = FlowState(intent="tu_application", entity=None, step=None)
        new = self.flow.next_state(ctx("подать заявку", state=state))
        assert new.step is None

    def test_non_progression_message_doesnt_advance(self):
        state = FlowState(intent="tu_application", entity="физическое лицо", step=3)
        new = self.flow.next_state(ctx("что такое мощность?", state=state))
        assert new.step == 3

    def test_original_state_not_mutated(self):
        state = FlowState(intent="tu_application", entity="физическое лицо", step=2)
        original_step = state.step
        self.flow.next_state(ctx("дальше", state=state))
        assert state.step == original_step  # model_copy() used — no mutation


class TestLinearFlowFinalStep:
    flow = LinearFlow()

    def test_final_step_all_progression_phrases(self):
        for word in ["дальше", "далее", "да", "следующий", "жалғастыр"]:
            state = FlowState(intent="tu_application", entity="физическое лицо", step=5)
            new = self.flow.next_state(ctx(word, state=state))
            assert new.step == 5, f"should stay at 5 for '{word}'"

    def test_final_step_non_progression_unchanged(self):
        state = FlowState(intent="tu_application", entity="юридическое лицо", step=5)
        new = self.flow.next_state(ctx("как подписать?", state=state))
        assert new.step == 5


class TestLinearFlowQuery:
    flow = LinearFlow()

    def test_query_contains_step_and_entity(self):
        state = FlowState(entity="физическое лицо", step=3)
        q = self.flow.build_query(ctx("что делать?", state=state))
        assert "шаг 3" in q
        assert "физическое лицо" in q

    def test_query_without_step_still_has_entity(self):
        state = FlowState(entity="юридическое лицо", step=None)
        q = self.flow.build_query(ctx("подать заявку", state=state))
        assert "юридическое лицо" in q
        assert "шаг" not in q

    def test_query_empty_entity_step_returns_message(self):
        state = FlowState(entity=None, step=None)
        q = self.flow.build_query(ctx("вопрос", state=state))
        assert "вопрос" in q


# ===========================================================================
# ScenarioFlow — query building
# ===========================================================================

class TestScenarioFlow:
    flow = ScenarioFlow()

    def test_query_includes_entity(self):
        state = FlowState(entity="юридическое лицо")
        q = self.flow.build_query(ctx("добавить объект", state=state))
        assert "юридическое лицо" in q

    def test_query_uses_original_question_as_anchor(self):
        state = FlowState(entity="физическое лицо",
                          original_question="жылжымайтын мүлікті қалай қосуға болады?")
        q = self.flow.build_query(ctx("ЖТ", state=state))
        assert "жылжымайтын мүлікті қалай қосуға болады?" in q
        assert "физическое лицо" in q

    def test_query_falls_back_to_message_without_original_question(self):
        state = FlowState(entity="физическое лицо")
        q = self.flow.build_query(ctx("добавить объект", state=state))
        assert "добавить объект" in q
        assert "физическое лицо" in q

    def test_no_history_uses_message_only(self):
        state = FlowState(entity="физическое лицо")
        q = self.flow.build_query(ctx("добавить объект", state=state, history=[]))
        assert "добавить объект" in q

    # --- next_state / situation detection ---

    def test_cadastre_keyword_sets_situation_cadastre(self):
        state = FlowState(intent="real_estate", original_question="объект косу")
        new = self.flow.next_state(ctx("кадастр арқылы", state=state))
        assert new.situation == "через кадастровый номер"

    def test_address_keyword_sets_situation_address(self):
        state = FlowState(intent="real_estate", original_question="объект косу")
        new = self.flow.next_state(ctx("адресный регистр", state=state))
        assert new.situation == "через адресный регистр"

    def test_situation_answer_appended_to_original_question(self):
        state = FlowState(intent="real_estate", original_question="объект косу")
        new = self.flow.next_state(ctx("кадастр арқылы", state=state))
        assert "объект косу" in new.original_question
        assert "кадастр арқылы" in new.original_question

    def test_situation_not_overwritten_once_set(self):
        state = FlowState(intent="real_estate", situation="через кадастровый номер")
        new = self.flow.next_state(ctx("адресный регистр", state=state))
        assert new.situation == "через кадастровый номер"  # must not switch

    def test_unknown_answer_preserves_none_situation(self):
        state = FlowState(intent="real_estate")
        new = self.flow.next_state(ctx("не знаю", state=state))
        assert new.situation is None

    def test_question_about_cadastre_does_not_set_situation(self):
        state = FlowState(intent="real_estate")
        new = self.flow.next_state(ctx("как получить кадастровый номер?", state=state))
        assert new.situation is None

    def test_question_about_cadastre_kz_does_not_set_situation(self):
        state = FlowState(intent="real_estate")
        new = self.flow.next_state(ctx("кадастрлық нөмірді қалай алуға болады?", state=state))
        assert new.situation is None


# ===========================================================================
# FAQFlow
# ===========================================================================

class TestFAQFlow:
    flow = FAQFlow()

    def test_query_is_raw_message(self):
        q = self.flow.build_query(ctx("как зарегистрироваться?"))
        assert q == "как зарегистрироваться?"

    def test_does_not_require_entity(self):
        assert self.flow.requires_entity is False


# ===========================================================================
# Registry — intent classification
# ===========================================================================

class TestClassifyIntent:
    def test_detect_tu_from_keyword(self):
        assert classify_intent("технические условия", None, None) == "tu_application"

    def test_detect_tu_from_space_padded_keyword(self):
        # " ту " requires spaces around
        assert classify_intent("хочу ту подать", None, None) == "tu_application"

    def test_detect_re_from_keyword(self):
        assert classify_intent("добавить объект недвижимости", None, None) == "real_estate"

    def test_no_keywords_returns_none(self):
        assert classify_intent("как зарегистрироваться", None, None) is None

    def test_sticky_preserves_tu_intent(self):
        result = classify_intent("физическое лицо", None, "tu_application")
        assert result == "tu_application"

    def test_sticky_preserves_re_intent(self):
        result = classify_intent("кадастровый номер", None, "real_estate")
        # "кадастр" is a real_estate keyword — not a switch, preserves intent
        result2 = classify_intent("что-то вообще не связанное", None, "real_estate")
        assert result2 == "real_estate"

    def test_explicit_switch_tu_to_re(self):
        result = classify_intent("хочу добавить объект недвижимости", None, "tu_application")
        assert result == "real_estate"

    def test_explicit_switch_re_to_tu(self):
        result = classify_intent("мне нужны технические условия", None, "real_estate")
        assert result == "tu_application"

    def test_page_based_tu_detection(self):
        assert classify_intent("как заполнить?", "шаг 1", None) == "tu_application"
        assert classify_intent("как заполнить?", "шаг 3", None) == "tu_application"

    def test_page_based_re_detection(self):
        assert classify_intent("помогите", "объект недвижимости", None) == "real_estate"

    def test_none_intent_no_page_no_keywords(self):
        assert classify_intent("привет", None, None) is None


class TestHasFlowKeywords:
    def test_tu_keywords(self):
        assert has_flow_keywords("технические условия")
        assert has_flow_keywords("техусловия")

    def test_re_keywords(self):
        assert has_flow_keywords("объект недвижимости")
        assert has_flow_keywords("кадастровый номер")
        assert not has_flow_keywords("кадастр")  # bare "кадастр" no longer triggers

    def test_no_keywords(self):
        assert not has_flow_keywords("как зарегистрироваться")
        assert not has_flow_keywords("не пришло письмо")
        assert not has_flow_keywords("забыл пароль")
        assert not has_flow_keywords("физическое лицо")


class TestHasFlowKeywordsCurrentIntent:
    """Phase 4: FAQ gate checks current intent only, not all intents."""

    def test_foreign_keyword_does_not_block_faq_gate(self):
        # "пломба" is a meter_sealing keyword — must NOT block FAQ gate in tu_application flow
        assert not has_flow_keywords("нужно снять пломбу", intent="tu_application")

    def test_current_intent_keyword_blocks_faq_gate(self):
        # tu_application keyword inside tu_application flow — must block FAQ gate
        assert has_flow_keywords("хочу уточнить технические условия", intent="tu_application")

    def test_re_keyword_in_re_flow(self):
        assert has_flow_keywords("добавить объект недвижимости", intent="real_estate")

    def test_re_keyword_does_not_block_tu_flow(self):
        # "кадастровый номер" is real_estate keyword — must not block FAQ gate in tu_application
        assert not has_flow_keywords("кадастровый номер", intent="tu_application")

    def test_no_intent_falls_back_to_all_intents(self):
        # Without intent param: backward-compatible, checks all
        assert has_flow_keywords("пломба")
        assert has_flow_keywords("кадастровый номер")

    def test_unknown_intent_returns_false(self):
        assert not has_flow_keywords("технические условия", intent="unknown_service")


class TestClassifyIntentKazakh:
    """Verify KZ inflected forms (post normalize_kz) are recognized."""

    def test_load_calc_kz_possessive_accusative(self):
        # "жуктемесін" → "жүктемесін" after normalize_kz
        assert classify_intent("жүктемесін есептеу", None, None) == "load_calculation"

    def test_load_calc_kz_reversed_word_order(self):
        assert classify_intent("есептеу жүктемесін", None, None) == "load_calculation"

    def test_load_calc_kz_accusative(self):
        assert classify_intent("жүктемені анықтау керек", None, None) == "load_calculation"

    def test_load_calc_kz_elektr_prefix(self):
        assert classify_intent("электр жүктемесін", None, None) == "load_calculation"

    def test_draft_design_kz_accusative(self):
        assert classify_intent("эскиздік жобаны қалай жасауға болады", None, None) == "draft_design"

    def test_construction_works_kz_partial(self):
        # "жумыстары" → "жұмыстары" after normalize_kz
        assert classify_intent("монтаж жұмыстары туралы", None, None) == "construction_works"

    def test_construction_works_kz_kurylys(self):
        assert classify_intent("құрылыс жұмыстары қашан басталады", None, None) == "construction_works"

    def test_supply_contract_kz_accusative(self):
        assert classify_intent("тұрмыстық шартты қалай жасасуға болады", None, None) == "supply_contract_residential"

    def test_supply_contract_non_residential_kz(self):
        assert classify_intent("тұрмыстық емес шарт жасасу керек", None, None) == "supply_contract_non_residential"

    def test_supply_contract_non_residential_ru(self):
        assert classify_intent("как заключить небытовой договор", None, None) == "supply_contract_non_residential"

    def test_supply_contract_residential_ru(self):
        assert classify_intent("как заключить бытовой договор", None, None) == "supply_contract_residential"

    def test_meter_sealing_kz_meter_accusative(self):
        assert classify_intent("есептеуіш аспапты орнату", None, None) == "meter_sealing"


class TestGetFlow:
    def test_tu_returns_linear(self):
        assert isinstance(get_flow("tu_application"), LinearFlow)

    def test_re_returns_scenario(self):
        assert isinstance(get_flow("real_estate"), ScenarioFlow)

    def test_none_returns_faq(self):
        assert isinstance(get_flow(None), FAQFlow)

    def test_unknown_intent_returns_faq(self):
        assert isinstance(get_flow("unknown_intent"), FAQFlow)


# ===========================================================================
# Invalid transitions
# ===========================================================================

class TestInvalidTransitions:
    flow = LinearFlow()

    def test_no_step_without_entity_even_after_progression(self):
        """Step should never start without an entity set."""
        state = FlowState(intent="tu_application", entity=None, step=None)
        for word in ["дальше", "да", "следующий"]:
            new = self.flow.next_state(ctx(word, state=state))
            assert new.step is None, f"step must remain None for '{word}' without entity"

    def test_step_doesnt_regress(self):
        """Non-progression messages must not decrement step."""
        state = FlowState(intent="tu_application", entity="физическое лицо", step=4)
        new = self.flow.next_state(ctx("объясни снова шаг 2", state=state))
        assert new.step == 4
