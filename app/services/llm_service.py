import httpx
import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

_INTENT_LABELS = {
    "real_estate": "Добавление объекта недвижимости",
    "tu_application": "Заявление на технические условия",
}

_PAGE_SITUATION_MAP = {
    "real_estate": [
        ("главная", "Ситуация 1 (главная страница ЛК)"),
        ("объект", "Ситуации 1–5"),
    ],
    "tu_application": [
        ("шаг 1", "Шаг 1 — выбор типа заявления"),
        ("шаг 2", "Шаг 2 — данные заявителя"),
        ("шаг 3", "Шаг 3 — опросный лист"),
        ("шаг 4", "Шаг 4 — данные объекта"),
        ("шаг 5", "Шаг 5 — подпись и отправка"),
    ],
}


async def ask_llm(
    question: str,
    context: str,
    page: str = None,
    language: str = "ru",
    history: str = "",
    intent: str = None,
    entity: str = None,
) -> str:
    is_kz = language == "kz"
    lang_instruction = "Язык: казахский." if is_kz else "Язык: русский."
    page_context = f"\n<page>{page}</page>" if page else ""
    history_tag = f"<history>\n{history}</history>" if history else ""

    intent_label = _INTENT_LABELS.get(intent or "", "не определён")
    entity_label = entity or "не определено"
    intent_context_block = f"""<intent_context>
Интент: {intent_label}
Тип пользователя: {entity_label}
Страница: {page or "не указана"}
</intent_context>"""

    page_situation_text = "не определена"
    if intent and page and intent in _PAGE_SITUATION_MAP:
        lower_page = page.lower()
        for kw, label in _PAGE_SITUATION_MAP[intent]:
            if kw in lower_page:
                page_situation_text = label
                break
    page_to_situation_block = f"""<page_to_situation>
{page or "не указана"} → {page_situation_text}
</page_to_situation>""" if intent else ""

    system_prompt = f"""<role>
Ты — официальный ИИ-ассистент платформы iSEL.
{lang_instruction}{page_context}
</role>

{history_tag}

{intent_context_block}

{page_to_situation_block}

<language_rule>
{"Язык пользователя: КАЗАХСКИЙ. Отвечай ИСКЛЮЧИТЕЛЬНО на казахском языке. Если <context> содержит русские инструкции — переведи их на казахский полностью перед ответом. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО отвечать на русском языке." if is_kz else "Язык пользователя: РУССКИЙ. Отвечай на русском языке."}
</language_rule>

<intent_handling>
ПОШАГОВЫЕ ИНТЕНТЫ (заявление на ТУ, технические условия):
- Показывай ОДИН шаг за раз
- Определяй текущий шаг по истории диалога
- В конце каждого шага (кроме последнего) спрашивай: {"Келесі қадамда не істеу керектігін білгіңіз келе ме?" if is_kz else "Хотите узнать что делать на следующем шаге?"}
- Если пользователь написал "да / дальше / следующий / иә / келесі" — показывай следующий шаг
- После последнего шага пиши: {"Өтініш жіберуге дайын. Күйі **Өтініштер** қойындысында көрінеді." if is_kz else "Заявление готово к отправке. Статус появится во вкладке **Обращения**."}

СИТУАЦИОННЫЕ ИНТЕНТЫ (объект недвижимости и другие):
- НЕ выдавай все ситуации сразу
- Задай уточняющий вопрос чтобы определить ситуацию пользователя
- Пример: "Вы добавляете объект через адресный регистр или через кадастровый номер?"
- Затем дай инструкцию только для выбранной ситуации

ОПРЕДЕЛЕНИЕ ФЛ/ЮЛ:
- Тип пользователя уже определён маршрутизатором и передан в <intent_context>
- НИКОГДА не спрашивай пользователя о его типе лица
- Используй значение из <intent_context> для выбора нужного шаблона
</intent_handling>

<rules>
1. Используй историю диалога для понимания контекста и текущего шага.
2. Найди в контексте ОДИН точный блок соответствующий текущему шагу или ситуации.
3. Воспроизведи шаблон ДОСЛОВНО — не перефразируй, не сокращай, не обобщай.
4. Никогда не смешивай шаблоны из разных ситуаций или шагов.
5. Если подходящего шаблона нет в контексте — ответь точно: "Не могу найти точный ответ на ваш вопрос. Рекомендую обратиться к оператору: **+7 775 990 15 36**". НИКОГДА не дополняй ответ из собственных знаний или обучающих данных.
6. Никогда не начинай ответ со слов "Шаблон ответа".
7. Не придумывай шаги которых нет в документе.
8. Если вопрос не связан с платформой iSEL — вежливо откажи.
9. Если в контексте только список без инструкций — игнорируй его.
10. Используй ТОЛЬКО блоки с нумерованными шагами или буллетами.
11. Никогда не выдавай несколько шагов или ситуаций в одном ответе.
</rules>

<formatting>
- Шаги нумеруй: 1. 2. 3.
- Каждый шаг с новой строки
- Кнопки платформы жирным: **Добавить**, **Далее**, **Отправить**, **Подписать**
- Разделы платформы жирным: **Объекты недвижимости**, **Обращения**, **Данные заявления**
- Важные термины жирным: **физическое лицо**, **юридическое лицо**, **кадастровый номер**, **ЭЦП**
- Телефон жирным: **+7 775 990 15 36**
- Между блоками пустая строка
- Предупреждения начинай с ⚠️
- Ссылки указывай полностью
</formatting>"""

    prompt = f"<context>\n{context}\n</context>\n\n<question>\n{question}\n</question>\n\n<answer>"

    if LLM_PROVIDER == "openai":
        return await _call_openai(system_prompt, prompt)
    else:
        return await _call_ollama(system_prompt, prompt)


async def _call_openai(system: str, prompt: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1000
                }
            )
            data = response.json()
            if response.status_code != 200:
                error_msg = data.get("error", {}).get("message", "Unknown error")
                return f"Ошибка OpenAI: {error_msg}"
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Ошибка LLM-сервиса: {str(e)}"


async def _call_ollama(system: str, prompt: str) -> str:
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                    "options": {"temperature": 0.0}
                }
            )
            data = response.json()
            return data.get("response", "Ошибка генерации.")
        except Exception as e:
            return f"Ошибка LLM-сервиса: {str(e)}"