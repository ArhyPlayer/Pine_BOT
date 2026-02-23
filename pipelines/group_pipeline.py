"""
Haystack-пайплайн для индексации и поиска сообщений групповых чатов.

Архитектура:
  indexing_pipeline  : OpenAIDocumentEmbedder -> DocumentWriter -> PineconeDocumentStore
  querying_pipeline  : OpenAITextEmbedder -> PineconeEmbeddingRetriever
  generator          : OpenAIChatGenerator (резюме сессии и ответы на упоминания)

Каждый Telegram-чат хранится в отдельном Pinecone namespace ("group_{chat_id}"),
что обеспечивает изоляцию данных между чатами и не требует фильтрации.
"""

import threading
from datetime import datetime
from typing import Dict, List

from loguru import logger

from haystack import Document, Pipeline
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from config import Config


class GroupChatPipeline:
    """
    Управляет Haystack-пайплайнами для группового чата.

    Для каждого чата (chat_id) лениво создаётся отдельный PineconeDocumentStore
    в namespace "group_{chat_id}". Это позволяет изолировать переписку чатов
    без сложных metadata-фильтров.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._api_key = Secret.from_token(config.openai_api_key)

        # Кэш document store + пайплайнов на chat_id
        self._stores: Dict[int, PineconeDocumentStore] = {}
        self._indexers: Dict[int, Pipeline] = {}
        self._queriers: Dict[int, Pipeline] = {}
        self._lock = threading.Lock()

        # Генератор для резюме и ответов (один, не зависит от чата)
        generator_kwargs: dict = dict(
            api_key=self._api_key,
            model=config.chat_model,
        )
        if config.openai_base_url:
            generator_kwargs["api_base_url"] = config.openai_base_url

        self._generator = OpenAIChatGenerator(**generator_kwargs)

    # ------------------------------------------------------------------
    # Private: store & pipeline factory
    # ------------------------------------------------------------------

    def _get_store(self, chat_id: int) -> PineconeDocumentStore:
        """Возвращает (или создаёт) PineconeDocumentStore для данного чата."""
        if chat_id not in self._stores:
            self._stores[chat_id] = PineconeDocumentStore(
                index=self._config.pinecone_index_name,
                namespace=f"group_{chat_id}",
                dimension=1536,
                metric="cosine",
            )
            logger.debug("PineconeDocumentStore создан для чата {}", chat_id)
        return self._stores[chat_id]

    def _get_indexer(self, chat_id: int) -> Pipeline:
        """Возвращает (или создаёт) indexing pipeline для данного чата."""
        if chat_id not in self._indexers:
            store = self._get_store(chat_id)
            embedder_kwargs: dict = dict(
                api_key=self._api_key,
                model=self._config.embedding_model,
            )
            if self._config.openai_base_url:
                embedder_kwargs["api_base_url"] = self._config.openai_base_url

            pipeline = Pipeline()
            pipeline.add_component("embedder", OpenAIDocumentEmbedder(**embedder_kwargs))
            pipeline.add_component(
                "writer",
                DocumentWriter(document_store=store, policy=DuplicatePolicy.OVERWRITE),
            )
            pipeline.connect("embedder", "writer")
            self._indexers[chat_id] = pipeline
            logger.debug("Indexing pipeline создан для чата {}", chat_id)
        return self._indexers[chat_id]

    def _get_querier(self, chat_id: int) -> Pipeline:
        """Возвращает (или создаёт) querying pipeline для данного чата."""
        if chat_id not in self._queriers:
            store = self._get_store(chat_id)
            embedder_kwargs: dict = dict(
                api_key=self._api_key,
                model=self._config.embedding_model,
            )
            if self._config.openai_base_url:
                embedder_kwargs["api_base_url"] = self._config.openai_base_url

            pipeline = Pipeline()
            pipeline.add_component("text_embedder", OpenAITextEmbedder(**embedder_kwargs))
            pipeline.add_component(
                "retriever",
                PineconeEmbeddingRetriever(document_store=store, top_k=25),
            )
            pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
            self._queriers[chat_id] = pipeline
            logger.debug("Querying pipeline создан для чата {}", chat_id)
        return self._queriers[chat_id]

    # ------------------------------------------------------------------
    # Public: indexing
    # ------------------------------------------------------------------

    def index_message(
        self,
        text: str,
        chat_id: int,
        user_id: int,
        username: str,
        timestamp: int,
    ) -> None:
        """
        Индексирует одно сообщение группового чата в Pinecone через Haystack pipeline.

        Вызывается в фоновом потоке — не блокирует обработку следующих сообщений.
        """
        doc = Document(
            content=text,
            meta={
                "chat_id": chat_id,
                "user_id": user_id,
                "username": username,
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                "type": "group_message",
            },
        )
        try:
            with self._lock:
                indexer = self._get_indexer(chat_id)
            indexer.run({"embedder": {"documents": [doc]}})
            logger.debug("Сообщение '{}...' от @{} в чате {} индексировано", text[:40], username, chat_id)
        except Exception as exc:
            logger.error("Ошибка индексации сообщения в чате {}: {}", chat_id, exc)

    def index_document_chunks(
        self,
        chunks: List[dict],
        chat_id: int,
    ) -> None:
        """
        Индексирует чанки документа в namespace группового чата.

        Каждый элемент chunks — dict с ключами:
          text, filename, chunk_index, page_no (опц.), headings (опц.), uploader
        """
        if not chunks:
            return
        docs = [
            Document(
                content=c["text"],
                meta={
                    "chat_id": chat_id,
                    "filename": c.get("filename", ""),
                    "chunk_index": c.get("chunk_index", 0),
                    "page_no": c.get("page_no") or 0,
                    "headings": c.get("headings", ""),
                    "uploader": c.get("uploader", ""),
                    "type": "doc_chunk",
                },
            )
            for c in chunks
            if c.get("text", "").strip()
        ]
        try:
            with self._lock:
                indexer = self._get_indexer(chat_id)
            indexer.run({"embedder": {"documents": docs}})
            logger.info(
                "Документ '{}' — {} чанков индексировано в чате {}",
                chunks[0].get("filename", "?"), len(docs), chat_id,
            )
        except Exception as exc:
            logger.error("Ошибка индексации документа в чате {}: {}", chat_id, exc)

    # ------------------------------------------------------------------
    # Public: retrieval
    # ------------------------------------------------------------------

    def query_context(
        self,
        question: str,
        chat_id: int,
        top_k: int = 20,
    ) -> List[Document]:
        """
        Возвращает top_k наиболее релевантных сообщений чата по семантическому сходству.
        """
        try:
            with self._lock:
                querier = self._get_querier(chat_id)
            result = querier.run(
                {
                    "text_embedder": {"text": question},
                    "retriever": {"top_k": top_k},
                }
            )
            docs = result.get("retriever", {}).get("documents", [])
            logger.debug("Найдено {} релевантных сообщений для запроса в чате {}", len(docs), chat_id)
            return docs
        except Exception as exc:
            logger.error("Ошибка поиска контекста в чате {}: {}", chat_id, exc)
            return []

    # ------------------------------------------------------------------
    # Public: generation
    # ------------------------------------------------------------------

    def summarize_session(self, messages: List[dict]) -> str:
        """
        Генерирует структурированное резюме диалога:
        — суть обсуждения
        — позиции сторон при споре + мнение бота
        — итоговое решение / договорённость
        — намеченные задачи / действия
        """
        if not messages:
            return "Диалог пуст — нечего резюмировать."

        dialog_lines = [
            f"{m.get('username', 'Неизвестный')} [{m.get('datetime', '')}]: {m['text']}"
            for m in messages
        ]
        dialog = "\n".join(dialog_lines)

        prompt = (
            "Ты — умный помощник для командной работы. "
            "Ниже представлен диалог из рабочего чата команды.\n\n"
            "Проанализируй диалог и выдай структурированный отчёт:\n\n"
            "1. **Тема обсуждения** — кратко о чём шёл разговор\n"
            "2. **Итог / Решение** — если было принято решение или договорённость, "
            "сформулируй его чётко и однозначно\n"
            "3. **Спор / Разногласия** (если были) — изложи позиции каждой стороны "
            "и выскажи своё мнение, кто прав и почему\n"
            "4. **Задачи и следующие шаги** — перечисли конкретные действия, "
            "если они были намечены, с указанием ответственных\n\n"
            f"=== Диалог ===\n{dialog}\n=== Конец диалога ===\n\n"
            "Отчёт:"
        )

        try:
            response = self._generator.run(messages=[ChatMessage.from_user(prompt)])
            reply = response["replies"][0]
            return getattr(reply, "text", None) or str(getattr(reply, "content", "")) or ""
        except Exception as exc:
            logger.error("Ошибка генерации резюме: {}", exc)
            return f"Не удалось сгенерировать резюме: {exc}"

    def summarize_document(self, chunks: List[str], filename: str) -> str:
        """Генерирует одно предложение-резюме загруженного документа."""
        sample = "\n\n".join(chunks[:15])[:3000]
        prompt = (
            f"Файл: {filename}\n\n"
            f"Содержимое (фрагмент):\n{sample}\n\n"
            "Дай ровно одно предложение — краткое резюме этого документа. "
            "Начни с «Этот документ» или «Документ содержит». "
            "Только одно предложение, без вводных слов."
        )
        try:
            response = self._generator.run(messages=[ChatMessage.from_user(prompt)])
            reply = response["replies"][0]
            return getattr(reply, "text", None) or str(getattr(reply, "content", "")) or "Документ успешно обработан."
        except Exception as exc:
            logger.error("Ошибка генерации резюме документа: {}", exc)
            return "Документ успешно обработан и сохранён."

    def answer_mention(self, question: str, context_docs: List[Document]) -> str:
        """
        Отвечает на вопрос/упоминание бота.

        Различает два типа контекста:
          - doc_chunk   — фрагмент загруженного документа (показывает filename/раздел)
          - group_message — сообщение из переписки (показывает автора и время)
        """
        if not context_docs:
            context_block = "Релевантная информация не найдена."
        else:
            doc_lines: List[str] = []
            msg_lines: List[str] = []
            for doc in context_docs:
                doc_type = doc.meta.get("type", "group_message")
                if doc_type == "doc_chunk":
                    filename = doc.meta.get("filename", "документ")
                    headings = doc.meta.get("headings", "")
                    page = doc.meta.get("page_no")
                    source = f"«{filename}»"
                    if headings:
                        source += f" / {headings}"
                    if page:
                        source += f" стр. {page}"
                    doc_lines.append(f"[Из документа {source}]\n{doc.content}")
                else:
                    username = doc.meta.get("username", "Неизвестный")
                    dt = doc.meta.get("datetime", "")
                    msg_lines.append(f"• {username} [{dt}]: {doc.content}")

            parts: List[str] = []
            if doc_lines:
                parts.append("--- Фрагменты документов ---")
                parts.extend(doc_lines)
            if msg_lines:
                parts.append("--- Сообщения из переписки ---")
                parts.extend(msg_lines)
            context_block = "\n\n".join(parts)

        prompt = (
            "Ты — помощник в рабочем командном чате. Тебя упомянули и задали вопрос.\n\n"
            "В контексте могут быть фрагменты загруженных документов и/или сообщения из переписки.\n\n"
            "Правила ответа:\n"
            "— Если вопрос касается документа — ответь, опираясь на фрагменты документа, "
            "укажи название файла и раздел\n"
            "— Если вопрос касается переписки — укажи, кто и когда это говорил\n"
            "— Дай своё заключение или рекомендацию на основе найденного\n"
            "— Если контекст не содержит ответа — честно скажи об этом\n"
            "— Отвечай по-русски, кратко и по делу\n\n"
            f"=== Контекст ===\n{context_block}\n=== Конец контекста ===\n\n"
            f"Вопрос: {question}\n\n"
            "Ответ:"
        )

        try:
            response = self._generator.run(messages=[ChatMessage.from_user(prompt)])
            reply = response["replies"][0]
            return getattr(reply, "text", None) or str(getattr(reply, "content", "")) or ""
        except Exception as exc:
            logger.error("Ошибка генерации ответа на упоминание: {}", exc)
            return f"Не удалось сформировать ответ: {exc}"
