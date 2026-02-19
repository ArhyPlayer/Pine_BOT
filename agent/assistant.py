"""
HaystackAgent — агент с инструментами и контекстом из Pinecone.

Собирает Agent один раз при warm_up() и переиспользует для всех запросов.
"""

from typing import Dict, List, Optional

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools.component_tool import ComponentTool
from haystack.utils import Secret
from openai import OpenAI

from config import Config
from .tools import DogFactTool, DogImageDescribeTool


class HaystackAgent:
    """
    Обёртка над Haystack Agent.

    Инкапсулирует создание генератора, инструментов и самого агента.
    Зависимости принимаются через конструктор (DI).
    """

    _SYSTEM_PROMPT: str = (
        "Ты — умный персональный помощник с долговременной памятью.\n\n"
        "Правила:\n"
        "- Используй только факты из предоставленного контекста памяти о пользователе.\n"
        "- Если нужной информации нет — честно скажи об этом.\n"
        "- Дружелюбный, полезный, ведёшь диалог как настоящий помощник.\n\n"
        "Инструменты:\n"
        "- dog_fact: случайный факт о собаках (когда пользователь явно просит).\n"
        "- dog_image_describe: фото собаки + описание породы (когда просят картинку/породу).\n\n"
        "Используй инструменты только когда запрос явно к этому подходит."
    )

    def __init__(self, config: Config, openai_client: OpenAI) -> None:
        self._config = config
        self._openai_client = openai_client
        self._agent: Agent | None = None

    def warm_up(self) -> None:
        """Собирает и прогревает агента. Вызывать один раз при старте."""
        self._agent = self._build()
        self._agent.warm_up()

    def reply(
        self,
        user_message: str,
        context_from_memory: str,
        history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Запускает агента с учётом контекста из Pinecone и истории сессии.

        Args:
            user_message: Текущее сообщение пользователя.
            context_from_memory: Отформатированный контекст (факты о пользователе).
            history: Краткосрочная история текущей сессии
                     (список {"role": "user"/"assistant", "content": str}).

        Returns:
            Текст ответа агента.
        """
        if self._agent is None:
            raise RuntimeError("Вызовите warm_up() перед первым использованием.")

        messages: List[ChatMessage] = []

        # Долговременный контекст из Pinecone
        if context_from_memory.strip():
            messages.append(
                ChatMessage.from_system(
                    "Контекст о пользователе из долговременной памяти:\n"
                    + context_from_memory.strip()
                )
            )

        # Краткосрочная история сессии (чередующиеся user/assistant сообщения)
        for turn in (history or []):
            if turn["role"] == "user":
                messages.append(ChatMessage.from_user(turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(ChatMessage.from_assistant(turn["content"]))

        messages.append(ChatMessage.from_user(user_message))

        result = self._agent.run(messages=messages)
        if not result.get("messages"):
            return "Не удалось сформировать ответ. Попробуйте ещё раз."
        last = result["messages"][-1]
        return getattr(last, "text", None) or str(last)

    def _build(self) -> Agent:
        generator = OpenAIChatGenerator(
            model=self._config.chat_model,
            api_key=Secret.from_token(self._config.openai_api_key),
            api_base_url=self._config.openai_base_url,
            generation_kwargs={"max_completion_tokens": 1000},
        )
        dog_fact_tool = ComponentTool(
            component=DogFactTool(),
            name="dog_fact",
            description="Получить случайный факт о собаках.",
        )
        dog_image_tool = ComponentTool(
            component=DogImageDescribeTool(openai_client=self._openai_client),
            name="dog_image_describe",
            description="Получить фото собаки и описание породы с краткой историей.",
        )
        return Agent(
            chat_generator=generator,
            tools=[dog_fact_tool, dog_image_tool],
            system_prompt=self._SYSTEM_PROMPT,
            exit_conditions=["text"],
            max_agent_steps=10,
        )
