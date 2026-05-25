"""Tests for ServiceRegistry — config loading, intent detection, service queries."""
import pytest
from app.config.service_registry import ServiceRegistry

registry = ServiceRegistry()

_EXPECTED_SERVICES = {
    "tu_application", "real_estate",
    "supply_contract_residential", "supply_contract_non_residential",
    "load_calculation", "draft_design", "construction_works", "meter_sealing",
    "primary_connection_residential", "primary_connection_nonresidential",
    "secondary_connection", "contract_termination",
    "grid_disconnection", "equipment_testing",
}


# ===========================================================================
# Singleton
# ===========================================================================

class TestSingleton:
    def test_same_instance_returned(self):
        r1 = ServiceRegistry()
        r2 = ServiceRegistry()
        assert r1 is r2

    def test_services_loaded_on_first_access(self):
        assert len(registry.list_services()) > 0


# ===========================================================================
# Service loading
# ===========================================================================

class TestServiceLoading:
    def test_all_services_present(self):
        assert set(registry.list_services().keys()) == _EXPECTED_SERVICES

    def test_unknown_service_returns_none(self):
        assert registry.get_service("nonexistent") is None

    def test_tu_application_is_linear_flow(self):
        assert registry.get_service("tu_application").flow.type == "linear"

    def test_tu_application_max_steps_5(self):
        assert registry.get_service("tu_application").flow.max_steps == 5

    def test_tu_application_requires_entity(self):
        assert registry.get_service("tu_application").flow.requires_entity is True

    def test_real_estate_is_scenario_flow(self):
        assert registry.get_service("real_estate").flow.type == "scenario"

    def test_real_estate_does_not_require_entity(self):
        assert registry.get_service("real_estate").flow.requires_entity is False

    def test_all_faq_services_have_correct_type(self):
        non_faq = {
            "tu_application", "real_estate",
            "primary_connection_residential", "primary_connection_nonresidential",
            "secondary_connection", "contract_termination",
            "grid_disconnection", "equipment_testing",
        }
        faq_services = _EXPECTED_SERVICES - non_faq
        for sid in faq_services:
            svc = registry.get_service(sid)
            assert svc.flow.type == "faq", f"{sid} must be faq"

    def test_get_flow_type_helper(self):
        assert registry.get_flow_type("tu_application") == "linear"
        assert registry.get_flow_type("real_estate") == "scenario"
        assert registry.get_flow_type("meter_sealing") == "faq"

    def test_get_flow_type_unknown_returns_faq(self):
        assert registry.get_flow_type("does_not_exist") == "faq"


# ===========================================================================
# Display names
# ===========================================================================

class TestDisplayNames:
    def test_tu_display_name_ru_contains_keyword(self):
        name = registry.get_display_name("tu_application", "ru")
        assert "условия" in name.lower() or "заявлени" in name.lower()

    def test_tu_display_name_kz_not_empty(self):
        name = registry.get_display_name("tu_application", "kz")
        assert name and name != "tu_application"

    def test_real_estate_ru_name(self):
        name = registry.get_display_name("real_estate", "ru")
        assert "недвижим" in name.lower() or "объект" in name.lower()

    def test_supply_residential_ru_contains_bytovoy(self):
        name = registry.get_display_name("supply_contract_residential", "ru")
        assert "бытовой" in name.lower()

    def test_supply_nonresidential_ru_contains_nebytovoy(self):
        name = registry.get_display_name("supply_contract_non_residential", "ru")
        assert "небытовой" in name.lower()

    def test_unknown_service_returns_service_id(self):
        assert registry.get_display_name("ghost_service", "ru") == "ghost_service"


# ===========================================================================
# Navigation paths
# ===========================================================================

class TestNavigationPaths:
    def test_tu_ru_path_mentions_uslugi(self):
        path = registry.get_navigation_path("tu_application", "ru")
        assert "Услуги" in path or "услуг" in path.lower()

    def test_tu_kz_path_mentions_qyzmettar(self):
        path = registry.get_navigation_path("tu_application", "kz")
        assert "Қызметтер" in path or "қызмет" in path.lower()

    def test_real_estate_ru_path_mentions_obekty(self):
        path = registry.get_navigation_path("real_estate", "ru")
        assert "Объекты недвижимости" in path or "недвижим" in path.lower()

    def test_navigation_path_unknown_service_empty(self):
        assert registry.get_navigation_path("nonexistent", "ru") == ""


# ===========================================================================
# detect_intent — filename patterns
# ===========================================================================

class TestDetectIntentFilename:
    def test_tu_application_zajavleniya_tu(self):
        # "Форма создание заявления ТУ.txt" — Phase 1 fix
        assert registry.detect_intent("", "Форма создание заявления ТУ.txt") == "tu_application"

    def test_tu_application_tehusloviya(self):
        assert registry.detect_intent("", "техусловия_образец.docx") == "tu_application"

    def test_real_estate_nedvizhim(self):
        assert registry.detect_intent("", "Добавление объекта недвижимости.txt") == "real_estate"

    def test_supply_residential_bytovoy(self):
        assert registry.detect_intent("", "Договор бытовой.txt") == "supply_contract_residential"

    def test_supply_nonresidential_nebytovoy(self):
        assert registry.detect_intent("", "Договор небытовой.txt") == "supply_contract_non_residential"

    def test_draft_design_eskizn(self):
        assert registry.detect_intent("", "Разработка Эскизного проекта.txt") == "draft_design"

    def test_construction_stroiteln(self):
        assert registry.detect_intent("", "Строительно-монтажные работы.txt") == "construction_works"

    def test_meter_sealing_ustanovka(self):
        assert registry.detect_intent("", "Установка пломбы.txt") == "meter_sealing"

    def test_meter_sealing_snyatie(self):
        assert registry.detect_intent("", "Снятие пломбы.txt") == "meter_sealing"

    def test_load_calculation_nagruzk(self):
        assert registry.detect_intent("", "Расчёт электрической нагрузки.txt") == "load_calculation"

    def test_secondary_connection_filename(self):
        assert registry.detect_intent("", "Вторичное подключение.txt") == "secondary_connection"

    def test_contract_termination_filename(self):
        assert registry.detect_intent("", "Расторжение договора.txt") == "contract_termination"

    def test_grid_disconnection_filename(self):
        assert registry.detect_intent("", "Отключение от электросетей.txt") == "grid_disconnection"

    def test_equipment_testing_filename(self):
        assert registry.detect_intent("", "Испытание, измерение электрооборудования.txt") == "equipment_testing"

    def test_general_questions_no_match(self):
        assert registry.detect_intent("", "Общие вопросы.txt") is None

    def test_empty_filename_and_text_returns_none(self):
        assert registry.detect_intent("", "") is None


# ===========================================================================
# detect_intent — content patterns
# ===========================================================================

class TestDetectIntentContent:
    def test_content_tu_application(self):
        assert registry.detect_intent("технические условия для присоединения к сети", "") == "tu_application"

    def test_content_real_estate(self):
        # Use nominative form to match pattern "объект недвижимости" exactly
        assert registry.detect_intent("объект недвижимости как добавить", "") == "real_estate"

    def test_content_meter_sealing(self):
        assert registry.detect_intent("пломбы на приборе учета", "") == "meter_sealing"

    def test_content_secondary_connection(self):
        assert registry.detect_intent("вторичное подключение к сети", "") == "secondary_connection"

    def test_content_contract_termination(self):
        assert registry.detect_intent("расторжение договора электроснабжения", "") == "contract_termination"

    def test_content_grid_disconnection(self):
        assert registry.detect_intent("отключение от электросетей", "") == "grid_disconnection"

    def test_content_equipment_testing(self):
        assert registry.detect_intent("испытание электрооборудования уровень напряжения", "") == "equipment_testing"


# ===========================================================================
# build_intent_keywords_map
# ===========================================================================

class TestBuildIntentKeywordsMap:
    def test_all_services_in_map(self):
        kw_map = registry.build_intent_keywords_map()
        assert _EXPECTED_SERVICES == set(kw_map.keys())

    def test_tu_keywords_include_ru_space_padded(self):
        keywords = registry.build_intent_keywords_map()["tu_application"]
        assert any(" ту " in kw for kw in keywords)

    def test_tu_keywords_include_kz_form(self):
        keywords = registry.build_intent_keywords_map()["tu_application"]
        assert any("тқ" in kw or "техникалық" in kw for kw in keywords)

    def test_meter_sealing_keywords_include_plomb(self):
        keywords = registry.build_intent_keywords_map()["meter_sealing"]
        assert any("пломб" in kw for kw in keywords)


# ===========================================================================
# detect_intent_from_page
# ===========================================================================

class TestDetectIntentFromPage:
    def test_tu_page_shag(self):
        assert registry.detect_intent_from_page("шаг 1") == "tu_application"

    def test_tu_page_tehusloviya(self):
        assert registry.detect_intent_from_page("технических условий заявление") == "tu_application"

    def test_re_page_nedvizhimost(self):
        assert registry.detect_intent_from_page("объект недвижимости профиль") == "real_estate"

    def test_unrelated_page_returns_none(self):
        assert registry.detect_intent_from_page("главная страница кабинет") is None
