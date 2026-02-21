"""
Менеджер долговременной памяти пользователей.

Сохраняет сообщения пользователя и чанки документов в Pinecone.
При обращении к агенту возвращает семантически близкие записи
(сообщения + фрагменты документов) в качестве контекста.
"""

import hashlib
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

    Типы записей:
      - "message"   — сырое сообщение пользователя
      - "doc_chunk" — фрагмент загруженного документа
      - "doc_index" — паспорт документа (для команды /memory)

    Все операции изолированы по namespace'у пользователя.
    Принимает зависимости через конструктор (DI).
    """

    # Фразы, при которых возвращаем все записи без порогового фильтра.
    _ALL_TRIGGERS: tuple[str, ...] = (
        "все предпочтени", "все мои предпочтени", "мои предпочтени",
        "напомни мне", "расскажи о мне", "что ты знаешь обо мне",
        "моя память", "мои интересы", "обо мне",
    )

    # Нижний порог косинусного сходства при обычном поиске.
    _SCORE_THRESHOLD = 0.30

    def __init__(
        self,
        pinecone_manager: PineconeManager,
        openai_client: OpenAI,
        config: Config,
    ) -> None:
        self._pm = pinecone_manager
        self._client = openai_client
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
        Возвращает семантически близкие записи из Pinecone.

        Включает как прошлые сообщения пользователя, так и фрагменты
        загруженных документов — отбирает по косинусному сходству.
        При триггерных фразах возвращает все записи без порогового фильтра.
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
            and m.get("metadata", {}).get("type") != "doc_index"  # паспорта не в контекст
        ][:top_k]

    def _retrieve_all(self, namespace: str, top_k: int) -> List[Dict[str, Any]]:
        logger.debug("Запрошены все записи — без фильтрации по релевантности")
        results = self._pm.query_by_text(
            text="пользователь написал сказал документ",
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )
        return [
            self._to_memory(m)
            for m in results.get("matches", [])
            if m.get("metadata", {}).get("type") != "doc_index"
        ]

    @staticmethod
    def _to_memory(match: Dict[str, Any]) -> Dict[str, Any]:
        meta = match.get("metadata", {})
        return {
            "text": meta.get("text", ""),
            "score": match.get("score", 0),
            "type": meta.get("type", "message"),
            "id": match.get("id", ""),
            "filename": meta.get("filename", ""),
            "page_no": meta.get("page_no"),
            "headings": meta.get("headings", ""),
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
        check_duplicates: bool = True,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Сохраняет текст в Pinecone.

        Args:
            check_duplicates: True — проверять косинусное сходство перед записью.
                              False — всегда записывать (для чанков документов).
            doc_id: Кастомный ID записи (если не указан — генерируется по timestamp).
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
        record_id = doc_id or f"{user_id}_{memory_type}_{timestamp}"
        try:
            return self._pm.smart_upsert_text(
                text=text,
                doc_id=record_id,
                metadata=meta,
                namespace=namespace,
                check_duplicates=check_duplicates,
                update_if_duplicate=True,
            )
        except Exception as exc:
            logger.error("Ошибка при сохранении в память: {}", exc)
            return {"action": "error", "error": str(exc)}

    def save_doc_index(
        self,
        user_id: int,
        filename: str,
        chunk_count: int,
    ) -> None:
        """
        Сохраняет/обновляет «паспорт» загруженного документа.

        Использует детерминированный ID на основе хэша имени файла,
        поэтому повторная загрузка того же файла обновит существующую запись.
        """
        doc_id = (
            f"{user_id}_doc_index_{hashlib.md5(filename.encode()).hexdigest()}"
        )
        self.save(
            user_id=user_id,
            text=f"Загружен документ: {filename}",
            memory_type="doc_index",
            metadata={"filename": filename, "chunk_count": chunk_count},
            check_duplicates=False,  # ID детерминированный — upsert сам обновит
            doc_id=doc_id,
        )

    # ------------------------------------------------------------------
    # Documents list
    # ------------------------------------------------------------------

    def list_indexed_documents(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Возвращает список проиндексированных документов пользователя.

        Ищет записи типа doc_index в namespace пользователя.
        """
        namespace = self.get_namespace(user_id)
        try:
            results = self._pm.query_by_text(
                text="загружен документ файл",
                top_k=50,
                namespace=namespace,
                filter_dict={"type": {"$eq": "doc_index"}},
                include_metadata=True,
            )
            docs = []
            seen = set()
            for m in results.get("matches", []):
                meta = m.get("metadata", {})
                fn = meta.get("filename", "unknown")
                if fn not in seen:
                    seen.add(fn)
                    docs.append({
                        "filename": fn,
                        "chunk_count": int(meta.get("chunk_count", 0)),
                    })
            return docs
        except Exception as exc:
            logger.error("Ошибка при получении списка документов: {}", exc)
            return []

    # ------------------------------------------------------------------
    # Format context for LLM
    # ------------------------------------------------------------------

    @staticmethod
    def format_for_context(memories: List[Dict[str, Any]]) -> str:
        """
        Форматирует найденные записи в текстовый блок для системного промпта.

        Разделяет записи на два раздела: фрагменты документов и сообщения
        пользователя — чтобы LLM понимал источник каждой информации.
        """
        if not memories:
            return ""

        doc_lines: List[str] = []
        msg_lines: List[str] = []

        for i, m in enumerate(memories, 1):
            if m["type"] == "doc_chunk":
                source_parts = []
                if m.get("filename"):
                    source_parts.append(m["filename"])
                if m.get("page_no") is not None:
                    source_parts.append(f"стр. {m['page_no']}")
                if m.get("headings"):
                    source_parts.append(m["headings"][:60])
                source = f"  [{', '.join(source_parts)}]" if source_parts else ""
                doc_lines.append(f"{i}. {m['text']}{source}")
            else:
                msg_lines.append(f"{i}. {m['text']}")

        parts: List[str] = []
        if doc_lines:
            parts.append("\n\n=== Из загруженных документов ===")
            parts.extend(doc_lines)
            parts.append("=================================")
        if msg_lines:
            parts.append("\n=== Прошлые сообщения пользователя ===")
            parts.extend(msg_lines)
            parts.append("=======================================\n")

        return "\n".join(parts)
