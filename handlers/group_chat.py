"""
Обработчики для групповых чатов.

Функциональность:
  /listen_start  — начать запись сессии диалога
  /listen_stop   — остановить запись, сохранить все сообщения в Pinecone
                   и получить резюме / итог диалога от нейросети
  Документы      — файлы из группы обрабатываются через Docling и индексируются
                   в namespace группы (group_{chat_id}), чтобы бот мог отвечать
                   на вопросы по их содержимому
  Все сообщения  — автоматически индексируются в Pinecone через Haystack pipeline
  @упоминание    — бот отвечает на вопрос, используя контекст всей переписки чата

ВАЖНО: чтобы бот получал ВСЕ сообщения группы (не только команды/упоминания),
необходимо отключить «Privacy Mode» в настройках бота через @BotFather
(Bot Settings → Group Privacy → Turn off).
"""

import os
import threading
from datetime import datetime
from typing import Dict, List, Optional

import telebot
from loguru import logger
from telebot import types

from config import Config
from documents import SUPPORTED_EXTENSIONS, download_telegram_file
from pipelines.group_pipeline import GroupChatPipeline


class GroupChatHandlers:
    """
    Регистрирует обработчики для групповых чатов.

    Изолирован от приватных чатов: все handlers имеют фильтр
    `m.chat.type in ("group", "supergroup")`.
    """

    def __init__(
        self,
        bot: telebot.TeleBot,
        pipeline: GroupChatPipeline,
        config: Config,
        ingestion=None,   # Optional[DoclingIngestionPipeline] — избегаем циклического импорта
    ) -> None:
        self._bot = bot
        self._pipeline = pipeline
        self._config = config
        self._ingestion = ingestion  # None — обработка документов недоступна

        # {chat_id: bool} — активна ли запись сессии
        self._listening: Dict[int, bool] = {}
        # {chat_id: list[dict]} — буфер сообщений текущей сессии
        self._sessions: Dict[int, List[dict]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self) -> None:
        """Регистрирует все обработчики группового чата."""
        b = self._bot

        b.register_message_handler(
            self._on_help,
            commands=["help"],
            func=lambda m: m.chat.type in ("group", "supergroup"),
        )
        b.register_message_handler(
            self._on_listen_start,
            commands=["listen_start"],
            func=lambda m: m.chat.type in ("group", "supergroup"),
        )
        b.register_message_handler(
            self._on_listen_stop,
            commands=["listen_stop"],
            func=lambda m: m.chat.type in ("group", "supergroup"),
        )
        b.register_message_handler(
            self._on_bot_question,
            commands=["bot_question"],
            func=lambda m: m.chat.type in ("group", "supergroup"),
        )
        b.register_message_handler(
            self._on_document,
            func=lambda m: m.chat.type in ("group", "supergroup"),
            content_types=["document"],
        )
        # Основной обработчик всех текстовых сообщений группы.
        # Должен быть зарегистрирован ПОСЛЕ команд, чтобы команды имели приоритет.
        b.register_message_handler(
            self._on_group_message,
            func=lambda m: m.chat.type in ("group", "supergroup") and bool(m.text),
            content_types=["text"],
        )

    # ------------------------------------------------------------------
    # Commands: /help  /listen_start  /listen_stop  /bot_question
    # ------------------------------------------------------------------

    def _on_help(self, message: types.Message) -> None:
        self._bot.reply_to(
            message,
            "Команды бота в этом чате:\n\n"
            "/listen_start — начать запись диалога\n"
            "  Бот фиксирует все сообщения в векторную базу.\n\n"
            "/listen_stop — остановить запись и получить резюме\n"
            "  Нейросеть проанализирует диалог: выделит суть, решения, задачи.\n\n"
            "/bot_question [вопрос] — задать боту вопрос по истории чата\n"
            "  Пример: /bot_question что мы решили по базе данных?\n\n"
            "Также можно упомянуть бота напрямую:\n"
            f"  @{self._config.bot_username or 'бот'} что мы решили по базе данных?",
        )

    def _on_bot_question(self, message: types.Message) -> None:
        """Обрабатывает /bot_question [вопрос] — альтернатива упоминанию @бот."""
        parts = message.text.split(maxsplit=1)
        # parts[0] — команда (/bot_question или /bot_question@username)
        if len(parts) < 2 or not parts[1].strip():
            self._bot.reply_to(
                message,
                "Задайте вопрос после команды.\n"
                "Пример: /bot_question что мы решили по базе данных?",
            )
            return

        question = parts[1].strip()
        chat_id = message.chat.id
        self._bot.send_chat_action(chat_id, "typing")
        threading.Thread(
            target=self._handle_mention,
            args=(message, question, chat_id),
            daemon=True,
        ).start()

    def _on_listen_start(self, message: types.Message) -> None:
        chat_id = message.chat.id
        initiator = _display_name(message.from_user)

        with self._lock:
            already_active = self._listening.get(chat_id, False)
            self._listening[chat_id] = True
            self._sessions[chat_id] = []

        if already_active:
            self._bot.reply_to(
                message,
                "🔄 Запись перезапущена. Предыдущий буфер очищен.\n\n"
                "Отправьте /listen_stop для завершения и получения резюме.",
            )
        else:
            self._bot.reply_to(
                message,
                f"🎙 Запись диалога начата по команде @{initiator}.\n\n"
                "Я фиксирую все сообщения чата и сохраняю их в векторную базу.\n"
                "Отправьте /listen_stop, чтобы завершить запись и получить резюме.",
            )

    def _on_listen_stop(self, message: types.Message) -> None:
        chat_id = message.chat.id

        with self._lock:
            was_listening = self._listening.get(chat_id, False)
            self._listening[chat_id] = False
            messages = list(self._sessions.get(chat_id, []))
            self._sessions[chat_id] = []

        if not was_listening:
            self._bot.reply_to(
                message,
                "⚠️ Запись не была активна.\n"
                "Используйте /listen_start, чтобы начать запись диалога.",
            )
            return

        if not messages:
            self._bot.reply_to(
                message,
                "⚠️ За время записи не поступило ни одного сообщения.",
            )
            return

        self._bot.reply_to(
            message,
            f"⏹ Запись остановлена. Зафиксировано сообщений: {len(messages)}.\n\n"
            "Анализирую диалог — подождите...",
        )

        threading.Thread(
            target=self._process_session,
            args=(chat_id, messages),
            daemon=True,
        ).start()

    def _process_session(self, chat_id: int, messages: List[dict]) -> None:
        """Фоновая задача: генерирует и отправляет резюме сессии."""
        try:
            summary = self._pipeline.summarize_session(messages)
            self._bot.send_message(
                chat_id,
                f"📊 Резюме диалога:\n\n{summary}",
            )
        except Exception as exc:
            logger.exception("Ошибка при генерации резюме сессии в чате {}", chat_id)
            self._bot.send_message(chat_id, f"❌ Ошибка при анализе диалога: {exc}")

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def _on_document(self, message: types.Message) -> None:
        """Обрабатывает файлы, загруженные в группу."""
        if self._ingestion is None:
            self._bot.reply_to(
                message,
                "Обработка документов недоступна: docling не установлен.\n"
                "Установите: pip install docling",
            )
            return

        doc = message.document
        filename = doc.file_name or f"document_{doc.file_id}"
        ext = os.path.splitext(filename)[1].lower()

        if ext not in SUPPORTED_EXTENSIONS:
            self._bot.reply_to(
                message,
                f"Формат «{ext}» не поддерживается.\n"
                f"Поддерживаемые форматы: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )
            return

        uploader = _display_name(message.from_user)
        self._bot.reply_to(
            message,
            f"📄 Файл «{filename}» получен от @{uploader}.\n"
            "Запускаю анализ и сохранение в базу знаний чата...",
        )

        threading.Thread(
            target=self._process_document_background,
            args=(message.chat.id, doc, filename, uploader),
            daemon=True,
        ).start()

    def _process_document_background(
        self,
        chat_id: int,
        document,
        filename: str,
        uploader: str,
    ) -> None:
        """Фоновая задача: скачать → Docling → индексировать в группу → резюме."""
        temp_path: Optional[str] = None
        try:
            temp_path, _ = download_telegram_file(self._bot, document)

            # Docling: конвертируем и чанкуем
            from docling.document_converter import DocumentConverter
            from docling.chunking import HybridChunker

            converter = DocumentConverter()
            dl_doc = converter.convert(source=temp_path).document
            chunker = HybridChunker()
            raw_chunks = list(chunker.chunk(dl_doc=dl_doc))
            logger.info("Docling: '{}' → {} чанков", filename, len(raw_chunks))

            # Формируем список чанков для GroupChatPipeline
            chunks_data: List[dict] = []
            chunk_texts: List[str] = []
            for i, chunk in enumerate(raw_chunks):
                text = chunker.serialize(chunk=chunk)
                if not text.strip():
                    continue

                page_no: Optional[int] = None
                headings: str = ""
                try:
                    page_no = chunk.meta.doc_items[0].prov[0].page_no
                except (AttributeError, IndexError):
                    pass
                try:
                    h = getattr(chunk.meta, "headings", None)
                    headings = " / ".join(h) if h else ""
                except Exception:
                    pass

                chunks_data.append({
                    "text": text,
                    "filename": filename,
                    "chunk_index": i,
                    "page_no": page_no,
                    "headings": headings,
                    "uploader": uploader,
                })
                chunk_texts.append(text)

            # Индексируем в namespace группы через Haystack pipeline
            self._pipeline.index_document_chunks(chunks_data, chat_id)

            self._bot.send_message(
                chat_id,
                f"✅ Документ «{filename}» изучен и сохранён в базу знаний чата "
                f"({len(chunk_texts)} фрагментов). Теперь можно задавать вопросы по нему.",
            )

            # Краткое резюме содержимого
            if chunk_texts:
                summary = self._pipeline.summarize_document(chunk_texts, filename)
                self._bot.send_message(chat_id, f"📋 {summary}")

        except ImportError:
            self._bot.send_message(
                chat_id,
                "Для обработки документов установите: pip install docling",
            )
        except Exception as exc:
            logger.exception("Ошибка при обработке документа '{}' в чате {}", filename, chat_id)
            self._bot.send_message(chat_id, f"❌ Не удалось обработать «{filename}»: {exc}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    # ------------------------------------------------------------------
    # All group messages
    # ------------------------------------------------------------------

    def _on_group_message(self, message: types.Message) -> None:
        """
        Обрабатывает каждое текстовое сообщение группы:
          1. Если бот упомянут — генерирует ответ по контексту чата
          2. Если активна запись сессии — добавляет в буфер
          3. Всегда — индексирует сообщение в Pinecone (фон)
        """
        chat_id = message.chat.id
        user = message.from_user
        user_id = user.id if user else 0
        username = _display_name(user)
        text = message.text or ""

        if not text:
            return

        # Реакция на упоминание бота
        if self._is_bot_mentioned(message):
            self._bot.send_chat_action(chat_id, "typing")
            threading.Thread(
                target=self._handle_mention,
                args=(message, text, chat_id),
                daemon=True,
            ).start()
            # Сообщение-упоминание тоже индексируем и добавляем в сессию

        # Добавление в буфер активной сессии
        with self._lock:
            if self._listening.get(chat_id, False):
                self._sessions[chat_id].append(
                    {
                        "user_id": user_id,
                        "username": username,
                        "text": text,
                        "timestamp": message.date,
                        "datetime": datetime.fromtimestamp(message.date).strftime(
                            "%H:%M:%S"
                        ),
                    }
                )

        # Всегда индексируем в Pinecone в фоне
        threading.Thread(
            target=self._pipeline.index_message,
            args=(text, chat_id, user_id, username, message.date),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Mention handling
    # ------------------------------------------------------------------

    def _is_bot_mentioned(self, message: types.Message) -> bool:
        """
        Возвращает True, если в сообщении есть упоминание (@username) бота
        или text_mention (для пользователей без username).
        """
        entities = message.entities or []
        bot_username = (self._config.bot_username or "").lower()

        for entity in entities:
            if entity.type == "mention" and message.text:
                mention = message.text[entity.offset: entity.offset + entity.length]
                # Сравниваем без символа @, без учёта регистра
                if bot_username and mention.lstrip("@").lower() == bot_username:
                    return True
            elif entity.type == "text_mention":
                # text_mention используется для пользователей без username;
                # у ботов username всегда есть, но на всякий случай проверяем
                if entity.user and entity.user.is_bot:
                    return True

        return False

    def _handle_mention(
        self,
        message: types.Message,
        text: str,
        chat_id: int,
    ) -> None:
        """Фоновая задача: поиск по контексту и генерация ответа на упоминание."""
        try:
            docs = self._pipeline.query_context(text, chat_id, top_k=20)
            reply = self._pipeline.answer_mention(text, docs)
            self._bot.reply_to(message, reply)
        except Exception as exc:
            logger.exception("Ошибка при ответе на упоминание в чате {}", chat_id)
            self._bot.reply_to(message, f"❌ Произошла ошибка при обработке запроса: {exc}")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _display_name(user: Optional[types.User]) -> str:
    """Возвращает username или имя пользователя для отображения."""
    if user is None:
        return "unknown"
    return user.username or user.first_name or f"user_{user.id}"
