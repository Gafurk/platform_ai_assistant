"""Rule-based response layer.

Checked BEFORE RAG and LLM on every request.
Keys are matched by longest-first substring search to prevent short keys
shadowing longer, more specific ones.

Two dictionaries:
  SYSTEM_COMMANDS — identity, greetings, handoff
  NAVIGATION_RULES — platform navigation paths (status, documents, refusals)
"""

# ---------------------------------------------------------------------------
# Identity / greetings / handoff
# ---------------------------------------------------------------------------

SYSTEM_COMMANDS: dict[str, dict] = {
    "оператор":        {"answer": None, "handoff": True},
    "человек":         {"answer": None, "handoff": True},
    "поддержка":       {"answer": None, "handoff": True},
    "выход":           {"answer": "Диалог завершён. Если понадоблюсь — обращайтесь!", "handoff": False},
    "кто ты":          {"answer": "Я — ИИ-ассистент платформы iSEL. Помогаю с вопросами по заполнению заявлений, добавлению объектов недвижимости и навигации по платформе.", "handoff": False},
    "кем ты являешься":{"answer": "Я — ИИ-ассистент платформы iSEL. Помогаю с вопросами по заполнению заявлений, добавлению объектов недвижимости и навигации по платформе.", "handoff": False},
    "ты бот":          {"answer": "Да, я ИИ-ассистент платформы iSEL. Чем могу помочь?", "handoff": False},
    "что ты умеешь":   {"answer": "Я могу помочь с:\n• Добавлением объекта недвижимости\n• Заполнением заявления на технические условия\n• Навигацией по платформе\n• Вопросами по статусам заявлений\n\nЕсли вопрос сложный — передам оператору.", "handoff": False},
    "расскажи анекдот":{"answer": "Я — рабочий ассистент платформы iSEL. Могу помочь с вопросами по платформе. Чем могу помочь?", "handoff": False},
    "напиши код":      {"answer": "Я — ассистент платформы iSEL и помогаю только с вопросами по платформе. Чем могу помочь?", "handoff": False},
    "привет":          {"answer": "Здравствуйте! Я — ИИ-ассистент платформы iSEL. Чем могу помочь?", "handoff": False},
    "здравствуйте":    {"answer": "Здравствуйте! Я — ИИ-ассистент платформы iSEL. Чем могу помочь?", "handoff": False},
    "салем":           {"answer": "Сәлем! Мен iSEL платформасының ИИ-көмекшісімін. Қалай көмектесе аламын?", "handoff": False},
    "спасибо":         {"answer": "Рад помочь! Если будут ещё вопросы — обращайтесь.", "handoff": False},
    "рахмет":          {"answer": "Ризашылығыңыз! Басқа сұрақтар болса — хабарласыңыз.", "handoff": False},
}

# ---------------------------------------------------------------------------
# Platform navigation — exact paths for common status / document queries.
# Longer phrases must come before shorter substrings (handled by sorted search).
# ---------------------------------------------------------------------------

_NAV_STATUS = (
    "Статус вашей заявки находится здесь:\n\n"
    "**Личный кабинет → Раздел Услуги → вкладка Обращения**\n\n"
    "В таблице вы увидите все ваши обращения и их текущий статус."
)

_NAV_REFUSAL = (
    "Мотивированный отказ находится здесь:\n\n"
    "**Личный кабинет → Раздел Услуги → вкладка Обращения**\n\n"
    "Найдите строку с вашей заявкой и нажмите иконку **Скачать** "
    "или кнопку **Просмотр** в этой строке."
)

_NAV_DOWNLOAD_TU = (
    "Готовые технические условия доступны здесь:\n\n"
    "**Личный кабинет → Раздел Услуги → вкладка Обращения**\n\n"
    "Когда заявка перейдёт в статус «Выполнено», в строке обращения "
    "появится иконка **Скачать** или кнопка **Просмотр**."
)

# Kazakh equivalents
_NAV_STATUS_KZ = (
    "Өтінішіңіздің мәртебесі мына жерде:\n\n"
    "**Жеке кабинет → Қызметтер бөлімі → Өтініштер қосымша беті**\n\n"
    "Кестеде барлық өтініштеріңіз бен олардың ағымдағы мәртебесі көрсетілген."
)

_NAV_REFUSAL_KZ = (
    "Дәлелді бас тарту мына жерде:\n\n"
    "**Жеке кабинет → Қызметтер бөлімі → Өтініштер қосымша беті**\n\n"
    "Өтінішіңіз бар жолды тауып, сол жолдағы **Жүктеу** немесе "
    "**Қарау** батырмасын басыңыз."
)

_NAV_DOWNLOAD_TU_KZ = (
    "Дайын техникалық шарттар мына жерде:\n\n"
    "**Жеке кабинет → Қызметтер бөлімі → Өтініштер қосымша беті**\n\n"
    "Өтініш «Орындалды» мәртебесіне ауысқан соң, жолда **Жүктеу** "
    "немесе **Қарау** батырмасы пайда болады."
)

NAVIGATION_RULES: dict[str, dict] = {
    # --- Refusal / rejection (RU) ---
    "мотивированный отказ":      {"answer": _NAV_REFUSAL, "handoff": False},
    "скачать отказ":             {"answer": _NAV_REFUSAL, "handoff": False},
    "получить отказ":            {"answer": _NAV_REFUSAL, "handoff": False},
    "где найти отказ":           {"answer": _NAV_REFUSAL, "handoff": False},

    # --- Application status (RU) ---
    "статус заявки":             {"answer": _NAV_STATUS, "handoff": False},
    "статус обращения":          {"answer": _NAV_STATUS, "handoff": False},
    "статус моей заявки":        {"answer": _NAV_STATUS, "handoff": False},
    "где моя заявка":            {"answer": _NAV_STATUS, "handoff": False},
    "проверить статус":          {"answer": _NAV_STATUS, "handoff": False},

    # --- Download TU (RU) ---
    "скачать технические условия": {"answer": _NAV_DOWNLOAD_TU, "handoff": False},
    "получить технические условия": {"answer": _NAV_DOWNLOAD_TU, "handoff": False},
    "готовые технические условия":  {"answer": _NAV_DOWNLOAD_TU, "handoff": False},
    "где скачать тУ":            {"answer": _NAV_DOWNLOAD_TU, "handoff": False},
    "скачать тУ":                {"answer": _NAV_DOWNLOAD_TU, "handoff": False},

    # --- Kazakh equivalents ---
    "дәлелді бас тарту":         {"answer": _NAV_REFUSAL_KZ, "handoff": False},
    "бас тартуды жүктеу":        {"answer": _NAV_REFUSAL_KZ, "handoff": False},
    "өтінішімнің мәртебесі":     {"answer": _NAV_STATUS_KZ, "handoff": False},
    "өтінішімді қайда":          {"answer": _NAV_STATUS_KZ, "handoff": False},
    "техникалық шарттарды жүктеу": {"answer": _NAV_DOWNLOAD_TU_KZ, "handoff": False},
    "дайын техникалық шарт":     {"answer": _NAV_DOWNLOAD_TU_KZ, "handoff": False},
}


def check_rule(message: str) -> tuple[str | None, bool]:
    msg = message.lower().strip()

    # Check both dicts; longest key wins in both — prevents short keys shadowing longer ones.
    combined = {**SYSTEM_COMMANDS, **NAVIGATION_RULES}
    for key in sorted(combined.keys(), key=len, reverse=True):
        if key in msg:
            val = combined[key]
            return val["answer"], val["handoff"]

    return None, False
