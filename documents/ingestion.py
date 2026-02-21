"""
Пайплайн индексации документов через Docling.

Обрабатывает PDF, DOCX и другие форматы: конвертирует текст в чанки,
сохраняет каждый чанк в Pinecone с метаданными (filename, chunk_index, page_no).
Генерирует одно предложение-резюме после успешной индексации.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional

from loguru import logger
from openai import OpenAI

from config import Config
from memory.manager import MemoryManager

# Поддерживаемые расширения файлов (Docling умеет их парсить)
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf",
    ".docx",
    ".pptx",
    ".html",
    ".htm",
    ".md",
    ".adoc",
    ".asciidoc",
})


class DoclingIngestionPipeline:
    """
    Принимает путь к документу, конвертирует через Docling,
    разбивает на семантические чанки и сохраняет в Pinecone.

    Зависимости принимаются через конструктор (DI).
    """

    def __init__(
        self,
        memory: MemoryManager,
        config: Config,
        openai_client: OpenAI,
    ) -> None:
        self._memory = memory
        self._config = config
        self._client = openai_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, file_path: str, filename: str, user_id: int) -> List[str]:
        """
        Конвертирует документ через Docling и сохраняет все чанки в Pinecone.

        Args:
            file_path: Путь к временному файлу на диске.
            filename:  Оригинальное имя файла (для метаданных).
            user_id:   Telegram user ID (для изоляции по namespace).

        Returns:
            Список текстов чанков (используется для генерации резюме).

        Raises:
            ImportError: Если docling не установлен.
        """
        try:
            from docling.document_converter import DocumentConverter
            from docling.chunking import HybridChunker
        except ImportError as exc:
            raise ImportError(
                "Для обработки документов установите: pip install docling"
            ) from exc

        logger.info("Docling: конвертируем '{}'", filename)
        converter = DocumentConverter()
        dl_doc = converter.convert(source=file_path).document

        chunker = HybridChunker()
        all_chunks = list(chunker.chunk(dl_doc=dl_doc))
        logger.info("Docling: '{}' → {} чанков", filename, len(all_chunks))

        saved_texts: List[str] = []
        for i, chunk in enumerate(all_chunks):
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

            self._memory.save(
                user_id=user_id,
                text=text,
                memory_type="doc_chunk",
                metadata={
                    "filename": filename,
                    "chunk_index": i,
                    "page_no": page_no,
                    "headings": headings,
                },
                check_duplicates=False,
            )
            saved_texts.append(text)
            logger.debug(
                "  чанк {}/{} сохранён (стр. {}, раздел: {})",
                i + 1, len(all_chunks), page_no or "?", headings[:60] or "—",
            )

        # Сохраняем «паспорт» документа для команды /memory
        self._memory.save_doc_index(
            user_id=user_id,
            filename=filename,
            chunk_count=len(saved_texts),
        )

        logger.success(
            "Файл '{}' — {} чанков сохранено в Pinecone",
            filename, len(saved_texts),
        )
        return saved_texts

    def summarize(self, chunks: List[str], filename: str) -> str:
        """
        Генерирует одно предложение-резюме содержимого документа.

        Берёт первые чанки (не более 3 000 символов) для краткого резюме.
        """
        sample = "\n\n".join(chunks[:15])[:3000]
        try:
            response = self._client.chat.completions.create(
                model=self._config.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты помощник, который составляет краткие резюме документов. "
                            "Отвечай строго одним предложением на русском языке."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Файл: {filename}\n\n"
                            f"Содержимое (фрагмент):\n{sample}\n\n"
                            "Дай ровно одно предложение — краткое резюме этого документа. "
                            "Начни с «Этот документ» или «Документ содержит». "
                            "Только одно предложение, без вводных слов."
                        ),
                    },
                ],
                max_completion_tokens=150,
            )
            return response.choices[0].message.content or "Документ успешно обработан."
        except Exception as exc:
            logger.error("Ошибка генерации резюме: {}", exc)
            return "Документ успешно обработан и сохранён в память."


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def download_telegram_file(bot, document) -> tuple[str, str]:
    """
    Скачивает файл из Telegram и сохраняет во временный файл.

    Returns:
        (temp_path, filename) — путь к временному файлу и имя файла.
    """
    filename = document.file_name or f"document_{document.file_id}"
    ext = Path(filename).suffix.lower() or ".bin"
    file_info = bot.get_file(document.file_id)
    file_bytes = bot.download_file(file_info.file_path)

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        tmp.write(file_bytes)
    finally:
        tmp.close()

    return tmp.name, filename
