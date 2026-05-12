from __future__ import annotations

import httpx
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

_INTENT_LABELS = {
    "real_estate": "Добавление объекта недвижимости",
    "tu_application": "Заявление на технические условия",
}

# ---------------------------------------------------------------------------
# Static prompt prefix — identical across requests so OpenAI can cache it.
# Dynamic sections (history, intent_context, step_control) are appended after.
# ---------------------------------------------------------------------------

_STATIC_PROMPT_RU = """<role>
Ты — официальный ИИ-ассистент платформы iSEL.
Язык: русский.
</role>

<language_rule>
Язык пользователя: РУССКИЙ. Отвечай на русском языке.
</language_rule>

<intent_handling>
ПОШАГОВЫЕ ИНТЕНТЫ (заявление на ТУ):
- Показывай ОДИН шаг за раз.
- Текущий шаг передаётся явно в <step_control>. Не определяй шаг самостоятельно.
- После шага 5 — не предлагай шаг 6 и не спрашивай о продолжении.

СИТУАЦИОННЫЕ ИНТЕНТЫ (объект недвижимости):
- Не выдавай все ситуации сразу.
- Задай уточняющий вопрос чтобы определить ситуацию пользователя.
- Пример: «Вы добавляете объект через адресный регистр или через кадастровый номер?»
- Затем дай инструкцию только для выбранной ситуации.

ОПРЕДЕЛЕНИЕ ФЛ/ЮЛ:
- Тип пользователя передаётся в <intent_context>.
- НИКОГДА не спрашивай пользователя о его типе лица повторно.
</intent_handling>

<rules>
1. Контекст извлечён из графа знаний платформы iSEL (LightRAG hybrid retrieval).
2. Используй историю диалога для понимания текущего шага или ситуации.
3. Найди в контексте ОДИН блок, соответствующий текущему шагу или ситуации.
4. Воспроизведи шаблон ДОСЛОВНО — не перефразируй, не сокращай.
5. Никогда не смешивай шаблоны из разных ситуаций или шагов.
6. Если подходящего шаблона нет — ответь точно: «Не могу найти точный ответ. Рекомендую обратиться к оператору: **+7 775 990 15 36**». Никогда не дополняй ответ из собственных знаний.
7. Никогда не начинай ответ со слов «Шаблон ответа» или «Граф показывает».
8. Не придумывай шаги, которых нет в документе.
9. Если вопрос не связан с платформой iSEL — вежливо откажи.
10. Если контекст содержит только список без инструкций — игнорируй его.
11. Используй ТОЛЬКО блоки с нумерованными шагами или буллетами.
12. Никогда не выдавай несколько шагов или ситуаций в одном ответе.
</rules>

<formatting>
- Шаги нумеруй: 1. 2. 3.
- Каждый шаг с новой строки.
- Кнопки платформы жирным: **Добавить**, **Далее**, **Отправить**, **Подписать**
- Разделы платформы жирным: **Объекты недвижимости**, **Обращения**, **Данные заявления**
- Важные термины жирным: **физическое лицо**, **юридическое лицо**, **кадастровый номер**, **ЭЦП**
- Телефон жирным: **+7 775 990 15 36**
- Между блоками пустая строка.
- Предупреждения начинай с ⚠️
- Ссылки указывай полностью.
</formatting>"""

_STATIC_PROMPT_KZ = """<role>
Ты — официальный ИИ-ассистент платформы iSEL.
Язык: казахский.
</role>

<language_rule>
Язык пользователя: КАЗАХСКИЙ. Отвечай ИСКЛЮЧИТЕЛЬНО на казахском языке.
Если <context> содержит русские инструкции — переведи их на казахский полностью перед ответом.
КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО отвечать на русском языке.
</language_rule>

<intent_handling>
ПОШАГОВЫЕ ИНТЕНТЫ (заявление на ТУ):
- Показывай ОДИН шаг за раз.
- Текущий шаг передаётся явно в <step_control>. Не определяй шаг самостоятельно.
- После шага 5 — не предлагай шаг 6 и не спрашивай о продолжении.

СИТУАЦИОННЫЕ ИНТЕНТЫ (объект недвижимости):
- Не выдавай все ситуации сразу.
- Задай уточняющий вопрос чтобы определить ситуацию пользователя.
- Затем дай инструкцию только для выбранной ситуации.

ОПРЕДЕЛЕНИЕ ФЛ/ЮЛ (ЖТ/ЗТ):
- Тип пользователя передаётся в <intent_context>.
- НИКОГДА не спрашивай пользователя о его типе лица повторно.
</intent_handling>

<rules>
1. Контекст извлечён из графа знаний платформы iSEL (LightRAG hybrid retrieval).
2. Используй историю диалога для понимания текущего шага или ситуации.
3. Найди в контексте ОДИН блок, соответствующий текущему шагу или ситуации.
4. Воспроизведи шаблон ДОСЛОВНО — переводя на казахский при необходимости.
5. Никогда не смешивай шаблоны из разных ситуаций или шагов.
6. Если подходящего шаблона нет — ответь: «Дәл жауап таба алмадым. Операторға хабарласыңыз: **+7 775 990 15 36**».
7. Никогда не начинай ответ со слов «Жауап үлгісі» или «Граф показывает».
8. Не придумывай шаги, которых нет в документе.
9. Если вопрос не связан с платформой iSEL — вежливо откажи на казахском.
10. Если контекст содержит только список без инструкций — игнорируй его.
11. Используй ТОЛЬКО блоки с нумерованными шагами или буллетами.
12. Никогда не выдавай несколько шагов или ситуаций в одном ответе.
</rules>

<formatting>
- Шаги нумеруй: 1. 2. 3.
- Каждый шаг с новой строки.
- Кнопки платформы жирным: **Қосу**, **Әрі қарай**, **Жіберу**, **Қол қою**
- Важные термины жирным: **жеке тұлға**, **заңды тұлға**, **кадастрлық нөмір**, **ЭЦҚ**
- Телефон жирным: **+7 775 990 15 36**
- Между блоками пустая строка.
- Предупреждения начинай с ⚠️
</formatting>"""


def _build_system_prompt(
    language: str,
    history: str,
    intent: Optional[str],
    entity: Optional[str],
    current_step: Optional[int],
    page: Optional[str],
) -> str:
    static = _STATIC_PROMPT_KZ if language == "kz" else _STATIC_PROMPT_RU
    dynamic_parts: list[str] = []

    if history:
        dynamic_parts.append(f"<history>\n{history}\n</history>")

    if intent or entity or page:
        intent_label = _INTENT_LABELS.get(intent or "", "не определён")
        dynamic_parts.append(
            f"<intent_context>\n"
            f"Интент: {intent_label}\n"
            f"Тип пользователя: {entity or 'не определено'}\n"
            f"Страница: {page or 'не указана'}\n"
            f"</intent_context>"
        )

    if current_step is not None:
        next_step = current_step + 1
        if current_step < 5:
            step_note = f"В конце ответа спроси: «Хотите узнать, что делать на шаге {next_step}?»"
        else:
            step_note = "Это ПОСЛЕДНИЙ шаг. Дай инструкцию по подписи и отправке. НИКОГДА не предлагай шаг 6."
        dynamic_parts.append(
            f"<step_control>\n"
            f"Текущий шаг: {current_step} из 5\n"
            f"Отвечай СТРОГО по шагу {current_step}. Не включай другие шаги.\n"
            f"{step_note}\n"
            f"</step_control>"
        )

    if dynamic_parts:
        return static + "\n\n" + "\n\n".join(dynamic_parts)
    return static


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def ask_llm(
    question: str,
    context: str,
    page: Optional[str] = None,
    language: str = "ru",
    history: str = "",
    intent: Optional[str] = None,
    entity: Optional[str] = None,
    current_step: Optional[int] = None,
) -> str:
    system = _build_system_prompt(
        language=language,
        history=history,
        intent=intent,
        entity=entity,
        current_step=current_step,
        page=page,
    )
    prompt = f"<context>\n{context}\n</context>\n\n<question>\n{question}\n</question>\n\n<answer>"

    if LLM_PROVIDER == "openai":
        return await _call_openai(system, prompt)
    return await _call_ollama(system, prompt)


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

async def _call_openai(system: str, prompt: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1000,
                },
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
                    "options": {"temperature": 0.0},
                },
            )
            data = response.json()
            return data.get("response", "Ошибка генерации.")
        except Exception as e:
            return f"Ошибка LLM-сервиса: {str(e)}"
