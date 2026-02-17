"""
Telegram бот-помощник с долговременной памятью на базе Pinecone.

Бот общается с пользователями, запоминает информацию о них и использует
эту информацию для персонализированных ответов.
"""

import os
import time
from datetime import datetime
from typing import List, Dict, Any
import telebot
from telebot import types
from openai import OpenAI
from dotenv import load_dotenv
from pinecone_manager import PineconeManager

# Загрузка переменных окружения
load_dotenv()

# Инициализация бота
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("Необходимо указать TELEGRAM_BOT_TOKEN в .env файле")

bot = telebot.TeleBot(BOT_TOKEN)

# Инициализация OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
CHAT_MODEL = os.getenv("CHAT_MODEL", "o4-mini-2025-04-16")

if not OPENAI_API_KEY:
    raise ValueError("Необходимо указать OPENAI_API_KEY в .env файле")

if OPENAI_BASE_URL:
    openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
else:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Инициализация менеджера Pinecone
try:
    pm = PineconeManager()
    print("✓ Pinecone менеджер инициализирован")
except Exception as e:
    print(f"✗ Ошибка инициализации Pinecone: {e}")
    raise

# Системный промпт для бота
SYSTEM_PROMPT = """Ты - умный персональный ассистент с долговременной памятью. 

Твои особенности:
- Ты запоминаешь всю важную информацию о пользователе из разговоров
- Ты используешь эту информацию для персонализированных ответов
- Ты дружелюбный, полезный и внимательный к деталям
- Ты общаешься естественно и по-человечески
- Если ты что-то помнишь о пользователе, используй это в разговоре

ВАЖНО:
- ВСЕГДА используй ТОЛЬКО информацию из предоставленного контекста памяти
- НИКОГДА не выдумывай и не галлюцинируй факты о пользователе
- Если информации нет в контексте памяти - честно скажи, что не помнишь или не знаешь
- Если пользователь спрашивает о предпочтениях, а контекст пуст - скажи, что пока не накопил достаточно информации

Когда тебе предоставляется контекст из памяти, используй ТОЛЬКО его для ответа.
Если в памяти есть противоречивая информация, уточни у пользователя, что актуально сейчас."""


def get_user_namespace(user_id: int) -> str:
    """Возвращает namespace для конкретного пользователя."""
    return f"user_{user_id}"


def retrieve_relevant_memories(
    user_id: int,
    query: str,
    top_k: int = 10,
    prefer_facts: bool = False
) -> List[Dict[str, Any]]:
    """
    Извлекает релевантные воспоминания о пользователе из Pinecone.
    
    Args:
        user_id: ID пользователя Telegram
        query: Текст запроса для поиска релевантной информации
        top_k: Количество воспоминаний для извлечения
        prefer_facts: Приоритет фактам (для запросов о предпочтениях)
        
    Returns:
        Список релевантных воспоминаний
    """
    try:
        namespace = get_user_namespace(user_id)
        all_memories = []
        query_lower = query.lower()
        
        # Проверяем, запрашивает ли пользователь ВСЕ факты/предпочтения
        wants_all_facts = any(phrase in query_lower for phrase in [
            'все предпочтени', 'все мои предпочтени', 'мои предпочтени',
            'напомни мне', 'расскажи о мне', 'что ты знаешь обо мне',
            'моя память', 'мои интересы', 'обо мне'
        ])
        
        # Если пользователь хочет все факты - возвращаем их без фильтрации по score
        if wants_all_facts:
            print("📋 Запрошены все факты - возвращаем без фильтрации по релевантности")
            fact_results = pm.query_by_text(
                text="предпочтения интересы хобби любимое",  # Общий запрос
                top_k=50,  # Больше результатов
                namespace=namespace,
                filter_dict={"type": "fact"},
                include_metadata=True
            )
            
            for match in fact_results.get('matches', []):
                # Без порога - берем все факты
                memory = {
                    'text': match.get('metadata', {}).get('text', ''),
                    'score': match.get('score', 0),
                    'type': match.get('metadata', {}).get('type', 'unknown'),
                    'id': match.get('id', '')
                }
                all_memories.append(memory)
            
            return all_memories[:top_k]
        
        # Если запрос о предпочтениях/фактах, сначала ищем факты
        if prefer_facts or any(word in query_lower for word in ['предпочтени', 'любим', 'нравится', 'хобби', 'работа', 'интерес']):
            # Поиск только среди фактов
            fact_results = pm.query_by_text(
                text=query,
                top_k=top_k,
                namespace=namespace,
                filter_dict={"type": "fact"},
                include_metadata=True
            )
            
            for match in fact_results.get('matches', []):
                if match.get('score', 0) > 0.25:  # Еще более низкий порог
                    memory = {
                        'text': match.get('metadata', {}).get('text', ''),
                        'score': match.get('score', 0),
                        'type': match.get('metadata', {}).get('type', 'unknown'),
                        'id': match.get('id', '')
                    }
                    all_memories.append(memory)
        
        # Общий поиск по всем типам памяти
        general_results = pm.query_by_text(
            text=query,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        
        for match in general_results.get('matches', []):
            if match.get('score', 0) > 0.5:  # Достаточно релевантные
                memory_id = match.get('id', '')
                # Избегаем дубликатов
                if not any(m['id'] == memory_id for m in all_memories):
                    memory = {
                        'text': match.get('metadata', {}).get('text', ''),
                        'score': match.get('score', 0),
                        'type': match.get('metadata', {}).get('type', 'unknown'),
                        'id': memory_id
                    }
                    all_memories.append(memory)
        
        # Сортируем по релевантности, приоритет фактам
        all_memories.sort(key=lambda x: (x['type'] == 'fact', x['score']), reverse=True)
        
        return all_memories[:top_k]
        
    except Exception as e:
        print(f"❌ Ошибка при извлечении воспоминаний: {e}")
        import traceback
        traceback.print_exc()
        return []


def save_to_memory(
    user_id: int,
    text: str,
    memory_type: str = "dialog",
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Сохраняет информацию в долговременную память пользователя.
    
    Args:
        user_id: ID пользователя Telegram
        text: Текст для сохранения
        memory_type: Тип памяти (dialog, fact, preference и т.д.)
        metadata: Дополнительные метаданные
        
    Returns:
        Результат операции сохранения
    """
    try:
        namespace = get_user_namespace(user_id)
        timestamp = int(time.time())
        doc_id = f"{user_id}_{memory_type}_{timestamp}"
        
        # Подготовка метаданных
        meta = metadata or {}
        meta.update({
            "user_id": user_id,
            "type": memory_type,
            "timestamp": timestamp,
            "datetime": datetime.now().isoformat()
        })
        
        # Умная запись с проверкой дубликатов
        result = pm.smart_upsert_text(
            text=text,
            doc_id=doc_id,
            metadata=meta,
            namespace=namespace,
            check_duplicates=True,
            update_if_duplicate=True  # Обновляем существующую информацию
        )
        
        return result
    except Exception as e:
        print(f"Ошибка при сохранении в память: {e}")
        return {"action": "error", "error": str(e)}


def format_memories_for_context(memories: List[Dict[str, Any]]) -> str:
    """Форматирует воспоминания для добавления в контекст промпта."""
    if not memories:
        return ""
    
    context = "\n\n=== Информация о пользователе из памяти ===\n"
    for i, memory in enumerate(memories, 1):
        context += f"{i}. {memory['text']}\n"
    context += "===========================================\n"
    
    return context


def extract_facts(user_message: str, bot_response: str) -> List[str]:
    """
    Извлекает факты о пользователе из диалога с помощью LLM.
    
    Args:
        user_message: Сообщение пользователя
        bot_response: Ответ бота
        
    Returns:
        Список извлеченных фактов
    """
    try:
        extraction_prompt = f"""Проанализируй диалог и извлеки ТОЛЬКО новые важные факты о пользователе.

Диалог:
Пользователь: {user_message}
Ассистент: {bot_response}

Извлекай ТОЛЬКО:
- Предпочтения (любимый цвет, еда, музыка и т.д.)
- Факты о личности (профессия, хобби, навыки)
- Важные детали (имя, возраст, место жительства)
- Планы и цели

НЕ извлекай:
- Общие вопросы без информации
- Команды боту
- Простую благодарность или приветствия

Формат ответа: список фактов, каждый с новой строки. Если фактов нет - напиши "НЕТ".

Пример:
Любимый цвет - синий
Работает программистом
Увлекается фотографией"""

        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Ты эксперт по извлечению структурированной информации. Будь точным и лаконичным."},
                {"role": "user", "content": extraction_prompt}
            ],
            max_completion_tokens=300
        )
        
        result = response.choices[0].message.content.strip()
        
        if result == "НЕТ" or not result:
            return []
        
        # Разбиваем на отдельные факты
        facts = [fact.strip() for fact in result.split('\n') if fact.strip() and fact.strip() != "НЕТ"]
        return facts
        
    except Exception as e:
        print(f"Ошибка при извлечении фактов: {e}")
        return []


def generate_response(user_message: str, context: str = "") -> str:
    """
    Генерирует ответ с помощью OpenAI.
    
    Args:
        user_message: Сообщение пользователя
        context: Контекст из памяти
        
    Returns:
        Ответ ассистента
    """
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        # Добавляем контекст, если есть
        if context:
            messages.append({
                "role": "system",
                "content": f"Контекст из предыдущих разговоров:\n{context}"
            })
        
        messages.append({"role": "user", "content": user_message})
        
        # Генерация ответа
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_completion_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Ошибка при генерации ответа: {e}")
        return "Извините, произошла ошибка при обработке вашего запроса. Попробуйте еще раз."


# ============================================================================
# Обработчики команд бота
# ============================================================================

@bot.message_handler(commands=['start'])
def handle_start(message):
    """Обработчик команды /start"""
    user_id = message.from_user.id
    user_name = message.from_user.first_name or "друг"
    
    welcome_text = f"""Привет, {user_name}! 👋

Я твой персональный ассистент с долговременной памятью. Я запоминаю всё, что ты мне рассказываешь, и использую эту информацию, чтобы быть максимально полезным.

🧠 Что я умею:
• Запоминать информацию о тебе (предпочтения, факты, детали)
• Использовать эту информацию в будущих разговорах
• Отвечать на вопросы с учетом контекста наших прошлых бесед
• Помогать с различными задачами

📝 Доступные команды:
/start - Показать это приветствие
/help - Помощь и информация
/memory - Показать статистику памяти
/clear - Очистить мою память о тебе
/forget [текст] - Удалить конкретную информацию

Просто напиши мне что угодно, и мы начнем общаться! 😊"""
    
    bot.reply_to(message, welcome_text)


@bot.message_handler(commands=['help'])
def handle_help(message):
    """Обработчик команды /help"""
    help_text = """📚 Справка по использованию бота

🔹 Основные возможности:
Я запоминаю всю важную информацию из наших разговоров и использую её для персонализированных ответов.

🔹 Команды:
/start - Начать работу с ботом
/help - Показать эту справку
/memory - Статистика моей памяти о тебе
/clear - Полностью очистить память (осторожно!)
/forget [текст] - Попросить забыть конкретную информацию

🔹 Как я работаю:
1. Ты пишешь мне сообщение
2. Я ищу в памяти релевантную информацию о тебе
3. Использую эту информацию для формирования ответа
4. Сохраняю новую информацию из нашего диалога
5. Автоматически избегаю дубликатов в памяти

🔹 Примеры использования:
• "Я люблю программировать на Python"
• "Мой любимый цвет - синий"
• "Напомни мне о моих предпочтениях"
• "Что ты знаешь обо мне?"

💡 Совет: Чем больше ты мне рассказываешь, тем полезнее я становлюсь!"""
    
    bot.reply_to(message, help_text)


@bot.message_handler(commands=['memory'])
def handle_memory(message):
    """Обработчик команды /memory - показывает статистику памяти"""
    user_id = message.from_user.id
    namespace = get_user_namespace(user_id)
    
    try:
        # Получаем статистику
        stats = pm.get_stats()
        namespaces = stats.get('namespaces', {})
        user_stats = namespaces.get(namespace, {})
        vector_count = user_stats.get('vector_count', 0)
        
        if vector_count == 0:
            bot.reply_to(
                message,
                "🧠 Моя память о тебе пока пуста.\n\nНачни общаться со мной, и я буду запоминать важную информацию!"
            )
        else:
            memory_text = f"🧠 Моя память о тебе:\n\n"
            memory_text += f"📊 Всего записей: {vector_count}\n\n"
            
            # Получаем ВСЕ факты о пользователе
            facts = retrieve_relevant_memories(
                user_id=user_id,
                query="все мои предпочтения",  # Триггер для получения всех фактов
                top_k=20,
                prefer_facts=True
            )
            
            # Фильтруем только факты
            fact_list = [m for m in facts if m['type'] == 'fact']
            
            if fact_list:
                memory_text += "📝 Факты о тебе:\n\n"
                for i, mem in enumerate(fact_list, 1):
                    memory_text += f"{i}. {mem['text']}\n"
            else:
                memory_text += "📝 Фактов пока не сохранено.\nРасскажи мне о себе, и я запомню!\n"
            
            memory_text += "\n💡 Используй /clear чтобы очистить память"
            
            bot.reply_to(message, memory_text)
    except Exception as e:
        print(f"❌ Ошибка при получении статистики: {e}")
        import traceback
        traceback.print_exc()
        bot.reply_to(
            message,
            f"Произошла ошибка при получении статистики: {str(e)}"
        )


@bot.message_handler(commands=['clear'])
def handle_clear(message):
    """Обработчик команды /clear - очищает память о пользователе"""
    user_id = message.from_user.id
    
    # Создаем inline клавиатуру для подтверждения
    markup = types.InlineKeyboardMarkup()
    markup.row(
        types.InlineKeyboardButton("✅ Да, очистить", callback_data=f"clear_confirm_{user_id}"),
        types.InlineKeyboardButton("❌ Отмена", callback_data="clear_cancel")
    )
    
    bot.reply_to(
        message,
        "⚠️ Вы уверены, что хотите очистить всю мою память о вас?\n\n"
        "Это действие нельзя отменить!",
        reply_markup=markup
    )


@bot.callback_query_handler(func=lambda call: call.data.startswith('clear_'))
def handle_clear_callback(call):
    """Обработчик подтверждения очистки памяти"""
    if call.data == "clear_cancel":
        bot.edit_message_text(
            "Отменено. Память сохранена.",
            call.message.chat.id,
            call.message.message_id
        )
        return
    
    if call.data.startswith("clear_confirm_"):
        user_id = int(call.data.split("_")[2])
        
        # Проверяем, что пользователь очищает свою собственную память
        if call.from_user.id != user_id:
            bot.answer_callback_query(call.id, "Ошибка: неверный пользователь")
            return
        
        try:
            namespace = get_user_namespace(user_id)
            pm.delete(delete_all=True, namespace=namespace)
            
            bot.edit_message_text(
                "✅ Память успешно очищена. Я забыл всё, что знал о тебе.\n\n"
                "Давай начнём заново! 😊",
                call.message.chat.id,
                call.message.message_id
            )
        except Exception as e:
            bot.edit_message_text(
                f"❌ Ошибка при очистке памяти: {str(e)}",
                call.message.chat.id,
                call.message.message_id
            )


@bot.message_handler(commands=['forget'])
def handle_forget(message):
    """Обработчик команды /forget - удаляет конкретную информацию"""
    # Извлекаем текст после команды
    parts = message.text.split(maxsplit=1)
    
    if len(parts) < 2:
        bot.reply_to(
            message,
            "❓ Использование: /forget [что забыть]\n\n"
            "Например: /forget мой любимый цвет"
        )
        return
    
    query = parts[1]
    user_id = message.from_user.id
    
    bot.reply_to(
        message,
        f"🔍 Ищу в памяти информацию о: '{query}'..."
    )
    
    # Ищем похожие воспоминания
    memories = retrieve_relevant_memories(user_id, query, top_k=5, prefer_facts=True)
    
    if not memories:
        bot.reply_to(
            message,
            "🤷 Не нашел в памяти ничего похожего на это."
        )
        return
    
    # Показываем найденное и предлагаем удалить
    response = "📋 Нашел следующую информацию:\n\n"
    for i, mem in enumerate(memories, 1):
        mem_type = "📌 Факт" if mem['type'] == 'fact' else "💬 Диалог"
        response += f"{i}. [{mem_type}] {mem['text'][:100]}...\n"
        response += f"   (релевантность: {mem['score']:.2f})\n\n"
    
    response += "⚠️ Функция выборочного удаления в разработке.\n"
    response += "Используйте /clear для полной очистки памяти."
    
    bot.reply_to(message, response)


@bot.message_handler(func=lambda message: True, content_types=['text'])
def handle_message(message):
    """Обработчик всех текстовых сообщений"""
    user_id = message.from_user.id
    user_name = message.from_user.first_name or "пользователь"
    user_message = message.text
    
    # Показываем индикатор "печатает..."
    bot.send_chat_action(message.chat.id, 'typing')
    
    try:
        # Шаг 1: Извлекаем релевантные воспоминания
        print(f"\n🔍 Поиск воспоминаний для запроса: '{user_message[:50]}...'")
        memories = retrieve_relevant_memories(user_id, user_message, top_k=10)
        
        if memories:
            print(f"✓ Найдено {len(memories)} релевантных воспоминаний:")
            for i, mem in enumerate(memories[:3], 1):
                print(f"  {i}. [{mem['type']}] {mem['text'][:80]}... (score: {mem['score']:.3f})")
        else:
            print("⚠️ Релевантных воспоминаний не найдено")
        
        context = format_memories_for_context(memories)
        
        # Шаг 2: Генерируем ответ с учетом контекста
        response = generate_response(user_message, context)
        
        # Шаг 3: Отправляем ответ пользователю
        bot.reply_to(message, response)
        
        # Шаг 4: Извлекаем и сохраняем факты о пользователе
        print(f"\n📝 Извлечение фактов из диалога...")
        facts = extract_facts(user_message, response)
        
        if facts:
            print(f"✓ Извлечено {len(facts)} фактов:")
            for fact in facts:
                print(f"  - {fact}")
                # Сохраняем каждый факт отдельно
                save_to_memory(
                    user_id=user_id,
                    text=fact,
                    memory_type="fact",
                    metadata={"username": user_name, "extracted": True}
                )
        else:
            print("⚠️ Фактов не извлечено")
        
    except Exception as e:
        print(f"❌ Ошибка при обработке сообщения: {e}")
        import traceback
        traceback.print_exc()
        bot.reply_to(
            message,
            "😔 Извините, произошла ошибка при обработке вашего сообщения. Попробуйте еще раз."
        )


# ============================================================================
# Запуск бота
# ============================================================================

def main():
    """Основная функция запуска бота"""
    print("=" * 50)
    print("🤖 Telegram бот-помощник запускается...")
    print("=" * 50)
    print(f"✓ Модель чата: {CHAT_MODEL}")
    print(f"✓ Pinecone индекс: {os.getenv('PINECONE_INDEX_NAME')}")
    print(f"✓ Модель эмбеддингов: {os.getenv('EMBEDDING_MODEL')}")
    print("=" * 50)
    print("🚀 Бот запущен и готов к работе!")
    print("Нажмите Ctrl+C для остановки")
    print("=" * 50)
    
    try:
        # Запускаем polling
        bot.infinity_polling(timeout=60, long_polling_timeout=60)
    except KeyboardInterrupt:
        print("\n👋 Бот остановлен пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        raise


if __name__ == "__main__":
    main()
