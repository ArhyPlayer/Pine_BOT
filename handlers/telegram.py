"""
Обработчики команд и сообщений Telegram-бота.

Регистрирует хендлеры и делегирует работу MemoryManager и HaystackAgent.
"""

from collections import deque
from typing import Dict, List, Optional

import telebot
from loguru import logger
from telebot import types
from openai import OpenAI

from config import Config
from memory.manager import MemoryManager
from agent.assistant import HaystackAgent

# Максимальное число сообщений (user + assistant) в краткосрочной памяти сессии.
# 20 = 10 пар «вопрос / ответ» — достаточно для связного диалога.
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
    ) -> None:
        self._bot = bot
        self._memory = memory
        self._config = config
        self._client = openai_client
        self._agent = haystack_agent
        # Краткосрочная история для каждого пользователя (текущая сессия).
        # Не сохраняется между перезапусками бота — только в RAM.
        self._histories: Dict[int, deque] = {}

    def register(self) -> None:
        """Регистрирует все обработчики команд и сообщений."""
        b = self._bot
        b.register_message_handler(self._on_start, commands=["start"])
        b.register_message_handler(self._on_help, commands=["help"])
        b.register_message_handler(self._on_memory, commands=["memory"])
        b.register_message_handler(self._on_clear, commands=["clear"])
        b.register_message_handler(self._on_forget, commands=["forget"])
        b.register_message_handler(self._on_text, func=lambda m: True, content_types=["text"])
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
            "• Помогать с различными задачами\n\n"
            "📝 Команды:\n"
            "/start — Это приветствие\n"
            "/help — Справка\n"
            "/memory — Показать что я о тебе знаю\n"
            "/clear — Очистить память\n"
            "/forget [текст] — Забыть конкретную информацию\n\n"
            "Просто напиши мне что угодно! 😊",
        )

    def _on_help(self, message: types.Message) -> None:
        self._bot.reply_to(
            message,
            "📚 Справка\n\n"
            "🔹 Основные возможности:\n"
            "Запоминаю важную информацию из разговоров и использую её в ответах.\n\n"
            "🔹 Команды:\n"
            "/start — Приветствие\n"
            "/help — Эта справка\n"
            "/memory — Что я знаю о тебе\n"
            "/clear — Полностью очистить память\n"
            "/forget [текст] — Забыть конкретную информацию\n\n"
            "🔹 Примеры:\n"
            "• «Я люблю программировать на Python»\n"
            "• «Мой любимый цвет — синий»\n"
            "• «Что ты знаешь обо мне?»\n"
            "• «Расскажи факт о собаках»\n"
            "• «Покажи картинку собаки и опиши породу»\n\n"
            "💡 Чем больше ты рассказываешь — тем полезнее я становлюсь!",
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
                    "🧠 Моя память о тебе пока пуста.\n\nНачни общаться — я запомню твои сообщения!",
                )
                return

            memories = self._memory.retrieve(
                user_id=user_id, query="что я говорил рассказывал писал", top_k=20
            )

            lines = [f"🧠 Моя память о тебе:\n\n📊 Всего записей: {vector_count}\n"]
            if memories:
                lines.append("💬 Запомненные сообщения:\n")
                lines.extend(f"{i}. {m['text'][:120]}" for i, m in enumerate(memories, 1))
            else:
                lines.append("💬 Сообщений пока нет. Начни общаться — я запомню!")
            lines.append("\n💡 /clear — очистить память")
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
            "⚠️ Вы уверены, что хотите очистить всю память?\n\nЭто действие нельзя отменить!",
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
                message, "❓ Использование: /forget [что забыть]\n\nНапример: /forget мой любимый цвет"
            )
            return

        query, user_id = parts[1], message.from_user.id
        self._bot.reply_to(message, f"🔍 Ищу: «{query}»...")

        memories = self._memory.retrieve(user_id, query, top_k=5, prefer_facts=True)
        if not memories:
            self._bot.reply_to(message, "🤷 Ничего похожего в памяти не найдено.")
            return

        lines = ["📋 Нашёл следующее:\n"]
        for i, mem in enumerate(memories, 1):
            label = "📌 Факт" if mem["type"] == "fact" else "💬 Диалог"
            lines.append(f"{i}. [{label}] {mem['text'][:100]}...")
            lines.append(f"   (релевантность: {mem['score']:.2f})\n")
        lines.append("⚠️ Выборочное удаление в разработке. Используй /clear для полной очистки.")
        self._bot.reply_to(message, "\n".join(lines))

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

            # 1. Долговременный контекст из Pinecone
            logger.debug("Поиск воспоминаний: «{}…»", user_message[:50])
            memories = self._memory.retrieve(user_id, user_message, top_k=10)
            if memories:
                logger.debug("Найдено {} воспоминаний:", len(memories))
                for i, m in enumerate(memories[:3], 1):
                    logger.debug("  {}. [{}] {}… (score: {:.3f})", i, m["type"], m["text"][:80], m["score"])
            else:
                logger.debug("Долговременных воспоминаний не найдено")

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
