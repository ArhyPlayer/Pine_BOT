"""
Обработчики команд и сообщений Telegram-бота.

Регистрирует хендлеры и делегирует работу MemoryManager, HaystackAgent
и DoclingIngestionPipeline.
"""

import os
import threading
from collections import deque
from typing import Dict, List, Optional

import telebot
from loguru import logger
from telebot import types
from openai import OpenAI

from config import Config
from memory.manager import MemoryManager
from agent.assistant import HaystackAgent
from documents import DoclingIngestionPipeline, SUPPORTED_EXTENSIONS, download_telegram_file

# Максимальное число сообщений (user + assistant) в краткосрочной памяти сессии.
_MAX_HISTORY_MESSAGES = 20


class BotHandlers:
    """
    Регистрирует все хендлеры в экземпляре telebot.TeleBot.
    Не обращается к глобальному состоянию — все зависимости через __init__.
    """

    def __init__(
        self,
        bot: telebot.TeleBot,
        memory: MemoryManager,
        config: Config,
        openai_client: OpenAI,
        haystack_agent: Optional[HaystackAgent] = None,
        ingestion_pipeline: Optional[DoclingIngestionPipeline] = None,
    ) -> None:
        self._bot = bot
        self._memory = memory
        self._config = config
        self._client = openai_client
        self._agent = haystack_agent
        self._ingestion = ingestion_pipeline
        # Краткосрочная история для каждого пользователя (RAM, текущая сессия).
        self._histories: Dict[int, deque] = {}

    def register(self) -> None:
        """Регистрирует все обработчики команд и сообщений."""
        b = self._bot
        b.register_message_handler(self._on_start, commands=["start"])
        b.register_message_handler(self._on_help, commands=["help"])
        b.register_message_handler(self._on_memory, commands=["memory"])
        b.register_message_handler(self._on_clear, commands=["clear"])
        b.register_message_handler(self._on_forget, commands=["forget"])
        b.register_message_handler(
            self._on_document,
            content_types=["document"],
        )
        b.register_message_handler(
            self._on_text,
            func=lambda m: True,
            content_types=["text"],
        )
        b.register_callback_query_handler(
            self._on_clear_callback, func=lambda c: c.data.startswith("clear_")
        )

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _on_start(self, message: types.Message) -> None:
        name = message.from_user.first_name or "друг"
        self._bot.reply_to(
            message,
            f"Привет, {name}! 👋\n\n"
            "Я твой персональный ассистент с долговременной памятью. Я запоминаю всё, "
            "что ты мне рассказываешь, и использую это, чтобы быть максимально полезным.\n\n"
            "🧠 Что я умею:\n"
            "• Запоминать информацию о тебе (предпочтения, факты, детали)\n"
            "• Использовать её в будущих разговорах\n"
            "• Помогать с различными задачами\n"
            "• Принимать и анализировать документы (PDF, DOCX, PPTX, HTML…)\n\n"
            "📝 Команды:\n"
            "/start — Это приветствие\n"
            "/help — Справка\n"
            "/memory — Показать что я о тебе знаю\n"
            "/clear — Очистить память\n"
            "/forget [текст] — Забыть конкретную информацию\n\n"
            "📄 Просто пришли мне файл — я его изучу и смогу отвечать на вопросы по нему!\n\n"
            "Просто напиши мне что угодно! 😊",
        )

    def _on_help(self, message: types.Message) -> None:
        ext_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        self._bot.reply_to(
            message,
            "📚 Справка\n\n"
            "🔹 Основные возможности:\n"
            "Запоминаю важную информацию из разговоров и использую её в ответах.\n\n"
            "🔹 Команды:\n"
            "/start — Приветствие\n"
            "/help — Эта справка\n"
            "/memory — Что я знаю о тебе и какие документы загружены\n"
            "/clear — Полностью очистить память (сообщения + документы)\n"
            "/forget [текст] — Забыть конкретную информацию\n\n"
            "🔹 Работа с документами:\n"
            f"Поддерживаемые форматы: {ext_list}\n"
            "Пришли файл — я проанализирую его, сохраню в память и дам краткое резюме. "
            "Затем можешь задавать вопросы по содержимому!\n\n"
            "🔹 Примеры текстовых запросов:\n"
            "• «Я люблю программировать на Python»\n"
            "• «Мой любимый цвет — синий»\n"
            "• «Что ты знаешь обо мне?»\n"
            "• «Расскажи факт о собаках»\n"
            "• «Покажи картинку собаки и опиши породу»\n\n"
            "💡 Чем больше ты рассказываешь и загружаешь — тем полезнее я становлюсь!",
        )

    def _on_memory(self, message: types.Message) -> None:
        user_id = message.from_user.id
        namespace = MemoryManager.get_namespace(user_id)
        try:
            stats = self._memory._pm.get_stats()
            vector_count = stats.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)

            if vector_count == 0:
                self._bot.reply_to(
                    message,
                    "🧠 Моя память о тебе пока пуста.\n\n"
                    "Начни общаться или загрузи документ — я запомню!",
                )
                return

            lines = [f"🧠 Моя память о тебе:\n\n📊 Всего записей: {vector_count}\n"]

            # Список загруженных документов
            docs = self._memory.list_indexed_documents(user_id)
            if docs:
                lines.append("📁 Загруженные документы:\n")
                for i, d in enumerate(docs, 1):
                    lines.append(f"  {i}. {d['filename']} ({d['chunk_count']} фрагментов)")
                lines.append("")

            # Последние сообщения
            memories = self._memory.retrieve(
                user_id=user_id, query="что я говорил рассказывал писал", top_k=15
            )
            msg_memories = [m for m in memories if m["type"] == "message"]
            if msg_memories:
                lines.append("💬 Запомненные сообщения:\n")
                lines.extend(
                    f"  {i}. {m['text'][:120]}"
                    for i, m in enumerate(msg_memories, 1)
                )

            if not docs and not msg_memories:
                lines.append("💬 Сообщений пока нет. Начни общаться — я запомню!")

            lines.append("\n💡 /clear — очистить всю память")
            self._bot.reply_to(message, "\n".join(lines))

        except Exception as exc:
            logger.exception("Ошибка при получении статистики памяти")
            self._bot.reply_to(message, f"Произошла ошибка: {exc}")

    def _on_clear(self, message: types.Message) -> None:
        user_id = message.from_user.id
        markup = types.InlineKeyboardMarkup()
        markup.row(
            types.InlineKeyboardButton("✅ Да, очистить", callback_data=f"clear_confirm_{user_id}"),
            types.InlineKeyboardButton("❌ Отмена", callback_data="clear_cancel"),
        )
        self._bot.reply_to(
            message,
            "⚠️ Вы уверены, что хотите очистить всю память?\n\n"
            "Будут удалены все сообщения и загруженные документы.\n"
            "Это действие нельзя отменить!",
            reply_markup=markup,
        )

    def _on_clear_callback(self, call: types.CallbackQuery) -> None:
        if call.data == "clear_cancel":
            self._bot.edit_message_text(
                "Отменено. Память сохранена.", call.message.chat.id, call.message.message_id
            )
            return

        if call.data.startswith("clear_confirm_"):
            user_id = int(call.data.split("_")[2])
            if call.from_user.id != user_id:
                self._bot.answer_callback_query(call.id, "Ошибка: неверный пользователь")
                return
            try:
                self._memory._pm.delete(
                    delete_all=True, namespace=MemoryManager.get_namespace(user_id)
                )
                self._histories.pop(user_id, None)
                self._bot.edit_message_text(
                    "✅ Память очищена. Давай начнём заново! 😊",
                    call.message.chat.id,
                    call.message.message_id,
                )
            except Exception as exc:
                self._bot.edit_message_text(
                    f"❌ Ошибка при очистке: {exc}",
                    call.message.chat.id,
                    call.message.message_id,
                )

    def _on_forget(self, message: types.Message) -> None:
        parts = message.text.split(maxsplit=1)
        if len(parts) < 2:
            self._bot.reply_to(
                message,
                "❓ Использование: /forget [что забыть]\n\nНапример: /forget мой любимый цвет",
            )
            return

        query, user_id = parts[1], message.from_user.id
        self._bot.reply_to(message, f"🔍 Ищу: «{query}»...")

        memories = self._memory.retrieve(user_id, query, top_k=5)
        if not memories:
            self._bot.reply_to(message, "🤷 Ничего похожего в памяти не найдено.")
            return

        lines = ["📋 Нашёл следующее:\n"]
        for i, mem in enumerate(memories, 1):
            if mem["type"] == "doc_chunk":
                label = f"📄 {mem.get('filename', 'документ')}"
            elif mem["type"] == "message":
                label = "💬 Сообщение"
            else:
                label = "📌 Запись"
            lines.append(f"{i}. [{label}] {mem['text'][:100]}...")
            lines.append(f"   (релевантность: {mem['score']:.2f})\n")
        lines.append("⚠️ Выборочное удаление в разработке. Используй /clear для полной очистки.")
        self._bot.reply_to(message, "\n".join(lines))

    # ------------------------------------------------------------------
    # Document handler
    # ------------------------------------------------------------------

    def _on_document(self, message: types.Message) -> None:
        """Обрабатывает входящие файлы — запускает ingestion pipeline в фоне."""
        if self._ingestion is None:
            self._bot.reply_to(
                message,
                "❌ Обработка документов недоступна: docling не установлен.\n"
                "Установите: pip install docling",
            )
            return

        doc = message.document
        filename = doc.file_name or f"document_{doc.file_id}"
        ext = os.path.splitext(filename)[1].lower()

        if ext not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            self._bot.reply_to(
                message,
                f"❌ Формат «{ext}» не поддерживается.\n\n"
                f"Поддерживаемые форматы: {supported}",
            )
            return

        self._bot.reply_to(
            message,
            "📄 Файл получен. Запускаю анализ и сохранение. "
            "Это может занять немного времени…",
        )

        chat_id = message.chat.id
        user_id = message.from_user.id

        thread = threading.Thread(
            target=self._process_document_background,
            args=(chat_id, user_id, doc, filename),
            daemon=True,
        )
        thread.start()

    def _process_document_background(
        self,
        chat_id: int,
        user_id: int,
        document,
        filename: str,
    ) -> None:
        """Фоновая задача: скачать → обработать Docling → сохранить → резюме."""
        temp_path: Optional[str] = None
        try:
            # 1. Скачиваем файл
            logger.info("Скачиваем файл '{}' для user {}", filename, user_id)
            temp_path, _ = download_telegram_file(self._bot, document)

            # 2. Индексация через Docling + сохранение в Pinecone
            chunks = self._ingestion.process(
                file_path=temp_path,
                filename=filename,
                user_id=user_id,
            )

            # 3. Сообщение об успехе
            self._bot.send_message(
                chat_id,
                "✅ Готово. Я изучил этот файл, теперь можем его обсудить.",
            )

            # 4. Краткое резюме (одно предложение)
            if chunks:
                summary = self._ingestion.summarize(chunks, filename)
                self._bot.send_message(chat_id, f"📋 {summary}")

        except ImportError as exc:
            logger.error("docling не установлен: {}", exc)
            self._bot.send_message(
                chat_id,
                "❌ Для обработки документов установите docling:\n"
                "pip install docling",
            )
        except Exception as exc:
            logger.exception("Ошибка при обработке документа '{}'", filename)
            self._bot.send_message(
                chat_id,
                f"❌ Не удалось обработать файл «{filename}»: {exc}",
            )
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    # ------------------------------------------------------------------
    # Text messages
    # ------------------------------------------------------------------

    def _on_text(self, message: types.Message) -> None:
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "пользователь"
        user_message = message.text

        self._bot.send_chat_action(message.chat.id, "typing")
        try:
            # 0. Краткосрочная история сессии (RAM)
            if user_id not in self._histories:
                self._histories[user_id] = deque(maxlen=_MAX_HISTORY_MESSAGES)
            history: List[Dict] = list(self._histories[user_id])

            # 1. Долговременный контекст из Pinecone (сообщения + документы)
            logger.debug("Поиск воспоминаний: «{}…»", user_message[:50])
            memories = self._memory.retrieve(user_id, user_message, top_k=10)
            if memories:
                logger.debug("Найдено {} записей:", len(memories))
                for i, m in enumerate(memories[:3], 1):
                    src = f" [{m['filename']}]" if m.get("filename") else ""
                    logger.debug(
                        "  {}. [{}]{} {}… (score: {:.3f})",
                        i, m["type"], src, m["text"][:70], m["score"],
                    )
            else:
                logger.debug("Долговременных записей не найдено")

            context = self._memory.format_for_context(memories)

            # 2. Ответ через Haystack-агента или резервную генерацию
            response = (
                self._agent.reply(user_message, context, history=history)
                if self._agent is not None
                else self._fallback(user_message, context, history=history)
            )

            self._bot.reply_to(message, response)

            # 3. Обновление краткосрочной истории сессии
            self._histories[user_id].append({"role": "user", "content": user_message})
            self._histories[user_id].append({"role": "assistant", "content": response})

            # 4. Сохранение сообщения пользователя в Pinecone
            result = self._memory.save(
                user_id=user_id,
                text=user_message,
                memory_type="message",
                metadata={"username": user_name},
            )
            logger.debug("Pinecone save → {}", result.get("action", "saved"))

        except Exception as exc:
            logger.exception("Ошибка при обработке сообщения")
            self._bot.reply_to(message, "😔 Произошла ошибка. Попробуйте ещё раз.")

    # ------------------------------------------------------------------
    # Fallback (without Haystack)
    # ------------------------------------------------------------------

    def _fallback(
        self,
        user_message: str,
        context: str,
        history: Optional[List[Dict]] = None,
    ) -> str:
        messages: List[Dict] = [{"role": "system", "content": self._config.SYSTEM_PROMPT}]
        if context:
            messages.append({"role": "system", "content": f"Контекст:\n{context}"})
        for turn in (history or []):
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})
        try:
            result = self._client.chat.completions.create(
                model=self._config.chat_model,
                messages=messages,
                max_completion_tokens=1000,
            )
            return result.choices[0].message.content or ""
        except Exception as exc:
            logger.error("Ошибка резервной генерации: {}", exc)
            return "Произошла ошибка. Попробуйте ещё раз."
