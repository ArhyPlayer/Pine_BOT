"""
Конфигурация приложения.

Весь доступ к переменным окружения и константам — только через класс Config.
"""

import os
from dotenv import load_dotenv


class Config:
    """
    Хранит настройки приложения, загружаемые из переменных окружения.

    Instantiate один раз при старте и передавайте в зависимости через конструктор.
    """

    # Системный промпт для резервной генерации ответа (без агента Haystack)
    SYSTEM_PROMPT: str = (
        "Ты — умный персональный ассистент с долговременной памятью.\n\n"
        "Твои особенности:\n"
        "- Ты запоминаешь всю важную информацию о пользователе из разговоров\n"
        "- Ты используешь эту информацию для персонализированных ответов\n"
        "- Ты дружелюбный, полезный и внимательный к деталям\n"
        "- Ты общаешься естественно и по-человечески\n\n"
        "ВАЖНО:\n"
        "- ВСЕГДА используй ТОЛЬКО информацию из предоставленного контекста памяти\n"
        "- НИКОГДА не выдумывай и не галлюцинируй факты о пользователе\n"
        "- Если информации нет в контексте памяти — честно скажи, что не помнишь\n"
        "- Если в памяти есть противоречивая информация, уточни у пользователя\n"
    )

    def __init__(self) -> None:
        load_dotenv()

        self.telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
        self.chat_model: str = os.getenv("CHAT_MODEL", "o4-mini-2025-04-16")
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "")

        self._validate()

    def _validate(self) -> None:
        """Выбрасывает ValueError, если обязательные параметры не заданы."""
        required = {
            "TELEGRAM_BOT_TOKEN": self.telegram_bot_token,
            "OPENAI_API_KEY": self.openai_api_key,
        }
        for name, value in required.items():
            if not value:
                raise ValueError(f"Необходимо указать {name} в .env файле")
