from __future__ import annotations

from typing import Optional

from app.flows.base import BaseFlow
from app.flows.faq import FAQFlow
from app.flows.linear import LinearFlow
from app.flows.scenario import ScenarioFlow

_FLOWS: dict[Optional[str], BaseFlow] = {
    "tu_application": LinearFlow(),
    "real_estate": ScenarioFlow(),
    # New service intents — FAQ-based until dedicated flows are implemented
    "supply_contract": FAQFlow(),
    "load_calculation": FAQFlow(),
    "draft_design": FAQFlow(),
    "construction_works": FAQFlow(),
    "meter_sealing": FAQFlow(),
    None: FAQFlow(),
}

# Keywords that indicate a specific intent — used for both detection and
# FAQ-interruption detection (message has NONE of these → treat as FAQ).
INTENT_KEYWORDS: dict[str, list[str]] = {
    "tu_application": [
        # Only TU-specific terms — no generic "submit application" phrases.
        # Generic phrases like "өтініш беру" / "подать заявление" caused false positives
        # (e.g. "пломбаны алуға өтініш беру" was misclassified as tu_application).
        " ту ", "технические условия", "техусловия",
        "заявление на ту", "заявку на ту", "заявка на ту",
        # Kazakh
        " тқ ", "техникалық шарттар",
    ],
    "real_estate": [
        "объект недвижимости", "добавить объект",
        "добавление объекта",
        "кадастровый номер", "через кадастр", "по кадастру",
        "адресный регистр", "адрестік регистр",
        "кадастр арқылы",
        "жылжымайтын мүлік", "жылжымайтын мүлікті",
        "мүлік қосу", "объект қосу", "профильге қосу",
    ],
    "supply_contract": [
        "договор бытовой", "бытовой договор", "договор электроснабжения",
        "договор небытовой", "небытовой договор", "заключить договор",
        "бытового договора", "небытового договора",
        # Kazakh
        "тұрмыстық шарт", "тұрмыстық емес шарт", "электрмен жабдықтау шарты",
    ],
    "load_calculation": [
        "расчет нагрузки", "расчёт нагрузки", "расчет электрической",
        "электрическая нагрузка",
        # Kazakh
        "жүктеме есебі", "жүктемені есептеу", "электр жүктемесін есептеу",
    ],
    "draft_design": [
        "эскизный проект", "разработка эскизного", "проект внешнего",
        # Kazakh — "жоба әзірлеу" alone is too generic ("develop a project")
        "эскиздік жоба", "эскиздік жобаны әзірлеу",
    ],
    "construction_works": [
        "строительно-монтажные", "строительно монтажные", " смр ",
        # Kazakh
        "құрылыс-монтаж", "құрылыс монтаждау жұмыстары",
    ],
    "meter_sealing": [
        "пломб", "установка пломбы", "снятие пломбы", "прибор учета",
        # Kazakh
        "пломбаны орнату", "пломбаны алу", "есептеуіш аспап",
    ],
}

_PAGE_INTENT_MAP: dict[str, list[str]] = {
    "tu_application": [
        "технических условий", "техусловия",
        "шаг 1", "шаг 2", "шаг 3", "шаг 4", "шаг 5",
    ],
    "real_estate": [
        "объект недвижимости", "объекты недвижимости",
        "добавление объекта",
    ],
}


def get_flow(intent: Optional[str]) -> BaseFlow:
    return _FLOWS.get(intent, _FLOWS[None])


def classify_intent(
    message: str,
    page: Optional[str],
    current_intent: Optional[str],
) -> Optional[str]:
    """Return the intent for this turn.

    Priority:
    1. Explicit keyword-based switch away from current intent.
    2. Preserve current intent (sticky) if no switch detected.
    3. Page-based detection for fresh sessions.
    4. None (FAQ) if nothing matches.
    """
    lower = " " + message.lower() + " "

    # Allow explicit cross-intent switch from keyword evidence
    for intent, keywords in INTENT_KEYWORDS.items():
        if intent != current_intent and any(kw in lower for kw in keywords):
            return intent

    # No switch detected — preserve current intent (sticky)
    if current_intent is not None:
        return current_intent

    # Page-based detection for new sessions
    if page:
        lower_page = page.lower()
        for intent, keywords in _PAGE_INTENT_MAP.items():
            if any(kw in lower_page for kw in keywords):
                return intent

    return None


def has_flow_keywords(message: str) -> bool:
    """True if the message contains keywords specific to any tracked intent."""
    lower = " " + message.lower() + " "
    return any(
        any(kw in lower for kw in keywords)
        for keywords in INTENT_KEYWORDS.values()
    )
