from __future__ import annotations

import httpx
import os
from typing import Optional
from dotenv import load_dotenv

from app.services.prompt_builder import build_system_prompt

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")


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
    situation: Optional[str] = None,
) -> str:
    system = build_system_prompt(
        language=language,
        history=history,
        intent=intent,
        entity=entity,
        current_step=current_step,
        page=page,
        situation=situation,
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
