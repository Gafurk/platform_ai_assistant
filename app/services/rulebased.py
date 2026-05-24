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
    "оператор":        {"answer": "По сложным вопросам обращайтесь к оператору: **+7 775 990 15 36**", "handoff": False},
    "человек":         {"answer": "По сложным вопросам обращайтесь к оператору: **+7 775 990 15 36**", "handoff": False},
    "поддержка":       {"answer": "По сложным вопросам обращайтесь к оператору: **+7 775 990 15 36**", "handoff": False},
    "выход":           {"answer": "Диалог завершён. Если понадоблюсь — обращайтесь!", "handoff": False},
    "кто ты":          {"answer": "Я — ИИ-ассистент платформы iSEL. Помогаю с вопросами по заполнению заявлений, добавлению объектов недвижимости и навигации по платформе.", "handoff": False},
    "кем ты являешься":{"answer": "Я — ИИ-ассистент платформы iSEL. Помогаю с вопросами по заполнению заявлений, добавлению объектов недвижимости и навигации по платформе.", "handoff": False},
    "ты бот":          {"answer": "Да, я ИИ-ассистент платформы iSEL. Чем могу помочь?", "handoff": False},
    "что ты умеешь":   {"answer": "Я могу помочь с:\n• Добавлением объекта недвижимости\n• Заполнением заявления на технические условия\n• Навигацией по платформе\n• Вопросами по статусам заявлений\n\nЕсли вопрос сложный — обратитесь в поддержку: **+7 775 990 15 36**", "handoff": False},
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
    """
    Return (answer, handoff) if the message matches a rule, else (None, False).

    Four passes in priority order:

    1. NAVIGATION_RULES — substring, longest-key-first.
       Navigation phrases are specific enough that substring presence anywhere
       in the message is intentional and always wins over greetings.

    2. IDENTITY_KEYS — word-sequence match (exact tokens), longest-key-first.
       Checked before OPERATOR_KEYS so "ты бот или человек" correctly returns
       the identity answer instead of the operator answer.

    3. OPERATOR_KEYS {"оператор", "человек", "поддержка"} — substring match.
       Substring (not token) is used intentionally to catch inflected Cyrillic
       forms: "оператором", "поддержки", etc.  Always fires regardless of
       message length — a user requesting a human agent is always valid.

    4. GREETING_KEYS {"привет", "здравствуйте", "салем", "спасибо", "рахмет",
       "выход"} — exact token match + content-word guard.
       Only fires when the message contains fewer than 2 "content" words
       (words not in SOCIAL_FILLER).  This prevents:
           "привет подскажи как пользоваться ЭЦП"  →  real question, skip rule
       while still catching:
           "привет"                                 →  greeting, answer immediately
           "спасибо за помощь"                      →  thanks (1 content word)
    """
    import re

    msg_lower = message.lower().strip()
    # Tokenize: strip punctuation, split on whitespace
    words = re.sub(r"[^\w\s]", " ", msg_lower).split()

    # ── 1. Navigation rules — substring, longest key wins ───────────────────
    for key in sorted(NAVIGATION_RULES.keys(), key=len, reverse=True):
        if key in msg_lower:
            val = NAVIGATION_RULES[key]
            return val["answer"], val["handoff"]

    # ── Classify SYSTEM_COMMANDS into buckets ────────────────────────────────
    OPERATOR_KEYS = {"оператор", "человек", "поддержка"}
    GREETING_KEYS = {"привет", "здравствуйте", "салем", "спасибо", "рахмет", "выход"}
    IDENTITY_KEYS = {k for k in SYSTEM_COMMANDS if k not in OPERATOR_KEYS and k not in GREETING_KEYS}

    # ── 2. Identity keys — word-sequence, longest key wins ──────────────────
    for key in sorted(IDENTITY_KEYS, key=len, reverse=True):
        key_tokens = key.split()
        for i in range(len(words) - len(key_tokens) + 1):
            if words[i: i + len(key_tokens)] == key_tokens:
                val = SYSTEM_COMMANDS[key]
                return val["answer"], val["handoff"]

    # ── 3. Operator keys — substring (handles inflected Cyrillic forms) ──────
    for key in sorted(OPERATOR_KEYS, key=len, reverse=True):
        if key in msg_lower:
            val = SYSTEM_COMMANDS[key]
            return val["answer"], val["handoff"]

    # ── 4. Greeting keys — exact token + content-word guard ─────────────────
    # Words that do not count as "real content" when deciding if the message
    # is a pure greeting or carries a substantive question.
    SOCIAL_FILLER = {
        "привет", "здравствуйте", "салем", "спасибо", "рахмет", "выход",
        "пожалуйста", "добрый", "добрая", "доброе", "день", "утро", "вечер",
        "ночь", "хорошо", "ладно", "ок", "окей", "за", "большое", "вам",
    }
    content_words = [w for w in words if w not in SOCIAL_FILLER]
    greeting_only = len(content_words) < 2  # < 2 real words → treat as pure greeting

    for key in sorted(GREETING_KEYS, key=len, reverse=True):
        if key in words and greeting_only:
            val = SYSTEM_COMMANDS[key]
            return val["answer"], val["handoff"]

    return None, False