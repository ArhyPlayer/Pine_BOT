"""
Инструменты агента — примеры работы с внешними API.

- DogFactTool: случайный факт о собаках (dog-api.kinduff.com)
- DogImageDescribeTool: случайное фото собаки (dog.ceo) + описание породы через OpenAI Vision
"""

import os
from typing import Optional

import requests
from dotenv import load_dotenv
from haystack import component
from openai import OpenAI

load_dotenv()

_DOG_FACTS_URL = "https://dog-api.kinduff.com/api/facts?number=1"
_DOG_IMAGE_URL = "https://dog.ceo/api/breeds/image/random"
_VISION_PROMPT = (
    "По этой фотографии собаки определи породу (на русском). "
    "Дай краткую предысторию: как появилась порода, для чего выводилась, "
    "2–4 предложения. Пиши только по делу, без вступления."
)


@component
class DogFactTool:
    """
    Получает случайный факт о собаках из бесплатного API.
    Используй, когда пользователь просит что-то интересное о собаках.
    """

    @component.output_types(result=str)
    def run(self, **kwargs) -> dict:
        try:
            r = requests.get(_DOG_FACTS_URL, timeout=10)
            r.raise_for_status()
            data = r.json()
            facts = data.get("facts")
            fact = facts[0] if isinstance(facts, list) and facts else str(data)
            return {"result": fact or "Не удалось получить факт."}
        except Exception as exc:
            return {"result": f"Ошибка при запросе факта о собаках: {exc}"}


@component
class DogImageDescribeTool:
    """
    Получает случайное фото собаки и описывает породу через OpenAI Vision.
    Используй, когда пользователь просит картинку собаки или хочет узнать породу.
    """

    def __init__(self, openai_client: Optional[OpenAI] = None) -> None:
        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL") or None,
            )

    @component.output_types(result=str)
    def run(self, **kwargs) -> dict:
        try:
            r = requests.get(_DOG_IMAGE_URL, timeout=10)
            r.raise_for_status()
            image_url = r.json().get("message", "")
            if not image_url:
                return {"result": "Не удалось получить ссылку на изображение."}

            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _VISION_PROMPT},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                max_tokens=400,
            )
            description = response.choices[0].message.content or "Не удалось получить описание."
            return {"result": f"Фото: {image_url}\n\n{description}"}
        except Exception as exc:
            return {"result": f"Ошибка при получении или описании фото: {exc}"}
