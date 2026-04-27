import httpx
import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")


async def ask_llm(question: str, context: str, page: str = None, language: str = "ru", history: str = "") -> str:
    lang_instruction = "Отвечай на русском языке." if language == "ru" else "Қазақ тілінде жауап бер."
    page_context = f"\nПользователь на странице: {page}." if page else ""
    history_block = f"\nИСТОРИЯ ДИАЛОГА:\n{history}" if history else ""

    system_prompt = f"""Ты — официальный ИИ-ассистент платформы iSEL.
{lang_instruction}{page_context}{history_block}

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
1. Используй историю диалога чтобы понять контекст текущего вопроса.
2. Найди в контексте ОДИН наиболее точный блок "Шаблон ответа" для данного вопроса.
3. Воспроизведи шаблон ДОСЛОВНО — не перефразируй, не сокращай, не обобщай.
4. Если в документе есть разные ситуации для ФЛ и ЮЛ — СНАЧАЛА спроси пользователя кто он. Спрашивай ТОЛЬКО ОДИН РАЗ.
5. Если пользователь уже ответил ФЛ или ЮЛ в истории — используй соответствующий шаблон СРАЗУ.
6. Никогда не смешивай шаблоны из разных ситуаций.
7. Если подходящего шаблона нет — скажи честно и предложи позвать оператора.
8. Никогда не начинай ответ со слов "Шаблон ответа".
9. Не придумывай шаги которых нет в документе.
10. Если вопрос пользователя не связан с платформой iSEL — вежливо откажи и предложи задать вопрос по платформе. Не отвечай на вопросы про погоду, математику, программирование и другие не связанные темы.

ПРАВИЛА ФОРМАТИРОВАНИЯ:
- Шаги нумеруй: 1. 2. 3.
- Каждый шаг с новой строки
- Названия кнопок выделяй жирным: **Добавить**, **Далее**, **Отправить**
- Названия разделов выделяй жирным: **Объекты недвижимости**, **Данные заявления**
- Важные термины: **физическое лицо**, **юридическое лицо**, **кадастровый номер**
- Телефон жирным: **+7 775 990 15 36**
- Между блоками пустая строка
- Предупреждения начинай с эмодзи"""

    prompt = f"КОНТЕКСТ:\n{context}\n\nВОПРОС:\n{question}\n\nОТВЕТ:"

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
                    "temperature": 0.2,
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
                    "options": {"temperature": 0.2}
                }
            )
            data = response.json()
            return data.get("response", "Ошибка генерации.")
        except Exception as e:
            return f"Ошибка LLM-сервиса: {str(e)}"