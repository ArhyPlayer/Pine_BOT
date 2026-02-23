"""
Точка входа Telegram-бота.

Собирает все зависимости, настраивает loguru и запускает polling.
"""

import sys

import telebot
from loguru import logger
from openai import OpenAI

from config import Config
from memory import PineconeManager, MemoryManager
from documents import DoclingIngestionPipeline
from handlers import BotHandlers
from handlers.group_chat import GroupChatHandlers
from pipelines import GroupChatPipeline

# ------------------------------------------------------------------
# Loguru configuration
# ------------------------------------------------------------------

logger.remove()  # убираем дефолтный handler
logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    ),
    level="DEBUG",
    colorize=True,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_openai_client(config: Config) -> OpenAI:
    if config.openai_base_url:
        return OpenAI(api_key=config.openai_api_key, base_url=config.openai_base_url)
    return OpenAI(api_key=config.openai_api_key)


def _try_build_agent(config: Config, openai_client: OpenAI):
    """Пытается создать HaystackAgent. При отсутствии haystack-ai возвращает None."""
    try:
        from agent import HaystackAgent
        agent = HaystackAgent(config=config, openai_client=openai_client)
        agent.warm_up()
        logger.success("Haystack-агент инициализирован")
        return agent
    except ImportError:
        logger.warning("haystack-ai не установлен — используется резервная генерация ответов")
        return None
    except Exception as exc:
        logger.warning("Haystack-агент недоступен ({}) — используется резервная генерация", exc)
        return None


def _try_build_group_pipeline(config: Config) -> GroupChatPipeline | None:
    """
    Создаёт GroupChatPipeline для обработки групповых чатов.

    Требует haystack-ai и pinecone-haystack. При отсутствии зависимостей
    или ошибке инициализации возвращает None — групповые функции будут недоступны.
    """
    try:
        import haystack_integrations  # noqa: F401 — проверяем наличие пакета
        pipeline = GroupChatPipeline(config=config)
        logger.success("GroupChatPipeline инициализирован")
        return pipeline
    except ImportError:
        logger.warning(
            "pinecone-haystack не установлен — функции групповых чатов недоступны. "
            "Установите: pip install pinecone-haystack"
        )
        return None
    except Exception as exc:
        logger.warning("GroupChatPipeline недоступен ({})", exc)
        return None


def _try_build_ingestion(
    memory: MemoryManager,
    config: Config,
    openai_client: OpenAI,
) -> DoclingIngestionPipeline | None:
    """
    Создаёт DoclingIngestionPipeline.

    Не падает при отсутствии docling — бот продолжит работу без обработки файлов.
    """
    try:
        import docling  # noqa: F401 — проверяем наличие пакета
        pipeline = DoclingIngestionPipeline(
            memory=memory,
            config=config,
            openai_client=openai_client,
        )
        logger.success("DoclingIngestionPipeline инициализирован")
        return pipeline
    except ImportError:
        logger.warning(
            "docling не установлен — обработка файлов недоступна. "
            "Для включения выполните: pip install docling"
        )
        return None
    except Exception as exc:
        logger.warning("DoclingIngestionPipeline недоступен ({})", exc)
        return None


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 50)
    logger.info("🤖 Telegram бот-помощник запускается...")
    logger.info("=" * 50)

    config = Config()
    logger.info("Модель чата:        {}", config.chat_model)
    logger.info("Pinecone индекс:    {}", config.pinecone_index_name)
    logger.info("Модель эмбеддингов: {}", config.embedding_model)

    openai_client = _build_openai_client(config)

    try:
        pm = PineconeManager()
        logger.success("Pinecone инициализирован")
    except Exception as exc:
        logger.error("Ошибка инициализации Pinecone: {}", exc)
        raise

    memory = MemoryManager(
        pinecone_manager=pm,
        openai_client=openai_client,
        config=config,
    )

    haystack_agent = _try_build_agent(config, openai_client)
    ingestion_pipeline = _try_build_ingestion(memory, config, openai_client)
    group_pipeline = _try_build_group_pipeline(config)

    bot = telebot.TeleBot(config.telegram_bot_token)

    # Автоматически определяем username бота, если не задан в .env
    if not config.bot_username:
        try:
            me = bot.get_me()
            config.bot_username = me.username or ""
            logger.info("Username бота: @{}", config.bot_username)
        except Exception as exc:
            logger.warning("Не удалось получить username бота: {}", exc)

    # Сначала регистрируем групповые хендлеры (имеют приоритет в группах)
    if group_pipeline is not None:
        GroupChatHandlers(
            bot=bot,
            pipeline=group_pipeline,
            config=config,
            ingestion=ingestion_pipeline,
        ).register()
        logger.success("GroupChatHandlers зарегистрированы")
    else:
        logger.warning("Групповые функции отключены (GroupChatPipeline не инициализирован)")

    # Приватные хендлеры регистрируются после групповых
    BotHandlers(
        bot=bot,
        memory=memory,
        config=config,
        openai_client=openai_client,
        haystack_agent=haystack_agent,
        ingestion_pipeline=ingestion_pipeline,
    ).register()

    logger.info("=" * 50)
    logger.success("🚀 Бот запущен! Нажмите Ctrl+C для остановки.")
    logger.info("=" * 50)

    try:
        bot.infinity_polling(timeout=60, long_polling_timeout=60)
    except KeyboardInterrupt:
        logger.info("👋 Бот остановлен")
    except Exception as exc:
        logger.critical("Критическая ошибка: {}", exc)
        raise


if __name__ == "__main__":
    main()
