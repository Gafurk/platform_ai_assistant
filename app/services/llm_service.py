# app/services/llm_service.py

import httpx
import os
import json
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

EXTERNAL_API_KEY = os.getenv("EXTERNAL_API_KEY", "") 

async def ask_llm(question: str, context: str, page: str = None, language: str = "ru") -> str:
    """Универсальная функция для работы с LLM (Ollama или внешнее API)."""
    
    lang_instruction = "Отвечай на русском языке." if language == "ru" else "Қазақ тілінде жауап бер."
    page_context = f"Пользователь на странице: {page}." if page else ""

    system_prompt = f"""Ты — официальный ИИ-ассистент платформы iSEL.
                    {lang_instruction}
                    {page_context}

                    ИНСТРУКЦИЯ ПО ИНТЕНТАМ:
                    В контексте ниже есть разделы 'ИНТЕНТ' и 'СИТУАЦИЯ'. Если вопрос совпадает с ними, 
                    используй строго 'Шаблон ответа' из этого контекста. Не выдумывай инструкции.
                    Если ответа нет — предложи позвать оператора."""

    prompt = f"КОНТЕКСТ:\n{context}\n\nВОПРОС:\n{question}\n\nОТВЕТ:"

    if LLM_PROVIDER == "ollama":
        return await _call_llm(system_prompt, prompt)
    else:
        return "Provider not implemented yet."

async def _call_llm(system: str, prompt: str) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
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
            return response.json().get("response", "Ошибка генерации.")
        except Exception as e:
            return f"Ошибка LLM-сервиса: {str(e)}"