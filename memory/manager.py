"""
Менеджер долговременной памяти пользователей.

Сохраняет каждое сообщение пользователя в Pinecone как есть (без LLM-обработки).
При обращении к агенту возвращает семантически близкие прошлые сообщения
в качестве контекста.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI

from config import Config
from .store import PineconeManager


class MemoryManager:
    """
    Управляет долговременной памятью пользователей в Pinecone.

    Стратегия хранения: каждое сообщение пользователя сохраняется как отдельная
    запись (type="message"). Дубликаты устраняются через косинусное сходство
    в PineconeManager.smart_upsert_text.

    Все операции изолированы по namespace'у пользователя.
    Принимает зависимости через конструктор (DI).
    """

    # Фразы, при которых возвращаем все сохранённые сообщения без фильтрации.
    _ALL_TRIGGERS: tuple[str, ...] = (
        "все предпочтени", "все мои предпочтени", "мои предпочтени",
        "напомни мне", "расскажи о мне", "что ты знаешь обо мне",
        "моя память", "мои интересы", "обо мне",
    )

    # Нижний порог косинусного сходства при обычном поиске.
    # Для сырых сообщений ставим ниже, чем для лаконичных фактов.
    _SCORE_THRESHOLD = 0.30

    def __init__(
        self,
        pinecone_manager: PineconeManager,
        openai_client: OpenAI,
        config: Config,
    ) -> None:
        self._pm = pinecone_manager
        self._client = openai_client  # оставляем для совместимости (не используется здесь)
        self._config = config

    # ------------------------------------------------------------------
    # Namespace
    # ------------------------------------------------------------------

    @staticmethod
    def get_namespace(user_id: int) -> str:
        return f"user_{user_id}"

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        user_id: int,
        query: str,
        top_k: int = 10,
        prefer_facts: bool = False,  # оставлен для обратной совместимости
    ) -> List[Dict[str, Any]]:
        """
        Возвращает семантически близкие прошлые сообщения пользователя из Pinecone.

        При триггерных фразах («расскажи о мне», «что ты знаешь») возвращает
        все записи без порогового фильтра.
        """
        namespace = self.get_namespace(user_id)

        if any(phrase in query.lower() for phrase in self._ALL_TRIGGERS):
            return self._retrieve_all(namespace, top_k=50)

        results = self._pm.query_by_text(
            text=query,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )
        return [
            self._to_memory(m)
            for m in results.get("matches", [])
            if m.get("score", 0) > self._SCORE_THRESHOLD
        ][:top_k]

    def _retrieve_all(self, namespace: str, top_k: int) -> List[Dict[str, Any]]:
        logger.debug("Запрошены все записи — возвращаем без фильтрации по релевантности")
        results = self._pm.query_by_text(
            text="пользователь написал сказал",
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )
        return [self._to_memory(m) for m in results.get("matches", [])]

    @staticmethod
    def _to_memory(match: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "text": match.get("metadata", {}).get("text", ""),
            "score": match.get("score", 0),
            "type": match.get("metadata", {}).get("type", "message"),
            "id": match.get("id", ""),
        }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        user_id: int,
        text: str,
        memory_type: str = "message",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Сохраняет текст в Pinecone с проверкой дубликатов по косинусному сходству.

        По умолчанию type='message' — сырое сообщение пользователя.
        """
        namespace = self.get_namespace(user_id)
        timestamp = int(time.time())
        meta = dict(metadata or {})
        meta.update(
            {
                "user_id": user_id,
                "type": memory_type,
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat(),
            }
        )
        try:
            return self._pm.smart_upsert_text(
                text=text,
                doc_id=f"{user_id}_{memory_type}_{timestamp}",
                metadata=meta,
                namespace=namespace,
                check_duplicates=True,
                update_if_duplicate=True,
            )
        except Exception as exc:
            logger.error("Ошибка при сохранении в память: {}", exc)
            return {"action": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Format
    # ------------------------------------------------------------------

    @staticmethod
    def format_for_context(memories: List[Dict[str, Any]]) -> str:
        """Форматирует найденные записи в текстовый блок для системного промпта."""
        if not memories:
            return ""
        lines = ["\n\n=== Прошлые сообщения пользователя из долговременной памяти ==="]
        lines.extend(f"{i}. {m['text']}" for i, m in enumerate(memories, 1))
        lines.append("=================================================================\n")
        return "\n".join(lines)
