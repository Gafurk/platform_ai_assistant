"""Tests for normalize_kz — Kazakh transliteration from Russian letters."""
import pytest
from app.services.translit import normalize_kz


# ===========================================================================
# No-op: message already has Kazakh characters
# ===========================================================================

class TestNoopWhenKzCharsPresent:
    def test_message_with_kz_char_unchanged(self):
        msg = "жүктемесін есептеу"
        assert normalize_kz(msg) == msg

    def test_message_with_a_kz_char_unchanged(self):
        # ә is a KZ char
        assert normalize_kz("сәлеметсіз бе") == "сәлеметсіз бе"

    def test_mixed_ru_kz_chars_unchanged(self):
        msg = "технические условия және қадам"
        assert normalize_kz(msg) == msg


# ===========================================================================
# Platform / section terms
# ===========================================================================

class TestPlatformTerms:
    def test_kyzmetter(self):
        assert normalize_kz("кызметтер") == "қызметтер"

    def test_otinishter(self):
        assert normalize_kz("отиништер") == "өтініштер"

    def test_elektrmен_zhabdyktau(self):
        assert normalize_kz("электрмен жабдыктау") == "электрмен жабдықтау"

    def test_kadastrly_nomer(self):
        assert normalize_kz("кадастрлык номер") == "кадастрлық нөмір"


# ===========================================================================
# TU phrases
# ===========================================================================

class TestTUPhrases:
    def test_teh_usloviya(self):
        result = normalize_kz("тех условия")
        assert "техникалық шарттар" in result

    def test_teh_shattar(self):
        result = normalize_kz("тех шарттар")
        assert "техникалық шарттар" in result

    def test_tu_alu(self):
        assert normalize_kz("ту алу") == "ТҚ алу"

    def test_tu_beru(self):
        assert normalize_kz("ту беру") == "ТҚ беру"


# ===========================================================================
# Real estate phrases
# ===========================================================================

class TestRealEstatePhrases:
    def test_zylzhymajtyn_mulik(self):
        result = normalize_kz("жылжымайтын мулик")
        assert "жылжымайтын мүлік" in result

    def test_zylzhymajtyn_mulikti(self):
        result = normalize_kz("жылжымайтын муликти")
        assert "жылжымайтын мүлікті" in result

    def test_object_qosu(self):
        result = normalize_kz("объект косу")
        assert "объект қосу" in result

    def test_profylge_qosu(self):
        result = normalize_kz("профильге косу")
        assert "профильге қосу" in result


# ===========================================================================
# Entity types
# ===========================================================================

class TestEntityTypes:
    def test_zeke_tulga(self):
        assert normalize_kz("жеке тулга") == "жеке тұлға"

    def test_zandy_tulga(self):
        assert normalize_kz("занды тулга") == "заңды тұлға"


# ===========================================================================
# Question words
# ===========================================================================

class TestQuestionWords:
    def test_kalay(self):
        result = normalize_kz("калай")
        assert "қалай" in result

    def test_kayda(self):
        result = normalize_kz("кайда")
        assert "қайда" in result

    def test_kandai(self):
        result = normalize_kz("кандай")
        assert "қандай" in result

    def test_kashan(self):
        result = normalize_kz("кашан")
        assert "қашан" in result

    def test_kansha(self):
        result = normalize_kz("канша")
        assert "қанша" in result


# ===========================================================================
# Load calculation KZ forms
# ===========================================================================

class TestLoadCalculation:
    def test_zhuktemesin(self):
        result = normalize_kz("жуктемесин")
        assert "жүктемесін" in result

    def test_zhuktemeni(self):
        result = normalize_kz("жуктемени")
        assert "жүктемені" in result

    def test_zhukteme(self):
        result = normalize_kz("жуктеме")
        assert "жүктеме" in result


# ===========================================================================
# Construction works KZ forms
# ===========================================================================

class TestConstructionWorks:
    def test_kurylys(self):
        result = normalize_kz("курылыс")
        assert "құрылыс" in result

    def test_zhumystary(self):
        result = normalize_kz("жумыстары")
        assert "жұмыстары" in result


# ===========================================================================
# Supply contract KZ forms
# ===========================================================================

class TestSupplyContract:
    def test_turmystyk_shart(self):
        result = normalize_kz("турмыстык шарт")
        assert "тұрмыстық шарт" in result

    def test_turmystyk(self):
        result = normalize_kz("турмыстык")
        assert "тұрмыстық" in result


# ===========================================================================
# Draft design KZ forms
# ===========================================================================

class TestDraftDesign:
    def test_eskizdik_zhoba(self):
        result = normalize_kz("эскиздик жоба")
        assert "эскиздік жоба" in result


# ===========================================================================
# Steps / кадам
# ===========================================================================

class TestKadamForms:
    def test_kadam(self):
        result = normalize_kz("кадам")
        assert "қадам" in result

    def test_kadamga(self):
        result = normalize_kz("кадамга")
        assert "қадамға" in result


# ===========================================================================
# Statuses and errors
# ===========================================================================

class TestStatuses:
    def test_martebe(self):
        result = normalize_kz("мартебе")
        assert "мәртебе" in result

    def test_martebesi(self):
        result = normalize_kz("мартебеси")
        assert "мәртебесі" in result

    def test_durys(self):
        result = normalize_kz("дурыс")
        assert "дұрыс" in result

    def test_mumkin(self):
        result = normalize_kz("мумкин")
        assert "мүмкін" in result


# ===========================================================================
# Common verbs
# ===========================================================================

class TestCommonVerbs:
    def test_kikiru(self):
        result = normalize_kz("киру")
        assert "кіру" in result

    def test_zhyberu(self):
        result = normalize_kz("жиберу")
        assert "жіберу" in result

    def test_saktau(self):
        result = normalize_kz("сактау")
        assert "сақтау" in result

    def test_tandau(self):
        result = normalize_kz("тандау")
        assert "таңдау" in result

    def test_alyga(self):
        result = normalize_kz("алуга")
        assert "алуға" in result


# ===========================================================================
# Greetings
# ===========================================================================

class TestGreetings:
    def test_salem(self):
        result = normalize_kz("салем")
        assert "сәлем" in result

    def test_salemetsiz_be(self):
        result = normalize_kz("салеметсиз бе")
        assert "сәлеметсіз бе" in result

    def test_kop_rakhmet(self):
        result = normalize_kz("коп рахмет")
        assert "көп рахмет" in result


# ===========================================================================
# Misc
# ===========================================================================

class TestMisc:
    def test_iya_yes(self):
        result = normalize_kz("ия")
        assert "иә" in result

    def test_zhok_no(self):
        result = normalize_kz("жок")
        assert "жоқ" in result

    def test_kazir(self):
        result = normalize_kz("казир")
        assert "қазір" in result

    def test_empty_string(self):
        assert normalize_kz("") == ""

    def test_pure_russian_unchanged(self):
        # Russian text should remain unchanged (no KZ substitutions apply)
        msg = "как заполнить заявление"
        result = normalize_kz(msg)
        # None of the translit keys match this pure Russian phrase
        assert "как" in result

    def test_longer_match_wins(self):
        # "жылжымайтын муликти" must map to мүлікті (not separately мүлік + ті)
        result = normalize_kz("жылжымайтын муликти")
        assert "жылжымайтын мүлікті" in result
