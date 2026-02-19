"""
Низкоуровневая работа с Pinecone: векторы, документы, поиск, дубликаты.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Пороги косинусного сходства:
#   ≥ HIGH  → дубликат (обновляем)
#   ≥ LOW   → похожий (сохраняем как новый)
#   < LOW   → новая информация
SIMILARITY_THRESHOLD_HIGH = 0.80
SIMILARITY_THRESHOLD_LOW = 0.75


class PineconeManager:
    """
    Управляет индексом Pinecone: запись, поиск, обновление, удаление.

    Все параметры можно задать явно или через переменные окружения.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        dimension: int = 1536,
    ) -> None:
        load_dotenv()

        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.dimension = dimension

        if not self.api_key:
            raise ValueError("Необходимо указать PINECONE_API_KEY в .env или при инициализации")
        if not self.index_name:
            raise ValueError("Необходимо указать PINECONE_INDEX_NAME в .env или при инициализации")

        self.pc = Pinecone(api_key=self.api_key)

        openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        openai_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
        if not openai_key:
            raise ValueError("Необходимо указать OPENAI_API_KEY в .env или при инициализации")

        self.openai_client = (
            OpenAI(api_key=openai_key, base_url=openai_url)
            if openai_url
            else OpenAI(api_key=openai_key)
        )

        self.index = None
        self._connect_to_index()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _connect_to_index(self) -> None:
        try:
            if self.index_name not in [idx.name for idx in self.pc.list_indexes()]:
                logger.warning("Индекс '{}' не найден — создаём новый", self.index_name)
                self.create_index()
            self.index = self.pc.Index(self.index_name)
            logger.success("Подключено к индексу '{}'", self.index_name)
        except Exception as exc:
            raise Exception(f"Ошибка при подключении к индексу: {exc}") from exc

    def create_index(
        self,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        dim = dimension or self.dimension
        try:
            self.pc.create_index(
                name=self.index_name,
                dimension=dim,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            logger.success("Индекс '{}' создан (dimension={})", self.index_name, dim)
        except Exception as exc:
            raise Exception(f"Ошибка при создании индекса: {exc}") from exc

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def get_embedding(self, text: str) -> List[float]:
        try:
            response = self.openai_client.embeddings.create(
                input=text, model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as exc:
            raise Exception(f"Ошибка при получении эмбеддинга: {exc}") from exc

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self.openai_client.embeddings.create(
                input=texts, model=self.embedding_model
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            raise Exception(f"Ошибка при получении эмбеддингов: {exc}") from exc

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_vectors(self, vectors: List[tuple], namespace: str = "") -> None:
        try:
            self.index.upsert(vectors=vectors, namespace=namespace)
            logger.debug("Записано {} векторов в namespace '{}'", len(vectors), namespace)
        except Exception as exc:
            raise Exception(f"Ошибка при записи векторов: {exc}") from exc

    def upsert_text(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: str = "",
    ) -> None:
        try:
            embedding = self.get_embedding(text)
            meta = dict(metadata or {})
            meta["text"] = text
            self.index.upsert(vectors=[(doc_id, embedding, meta)], namespace=namespace)
            logger.debug("Документ '{}' записан", doc_id)
        except Exception as exc:
            raise Exception(f"Ошибка при записи текста: {exc}") from exc

    def upsert_documents(
        self,
        documents: List[Dict[str, Any]],
        id_field: str = "id",
        text_field: str = "text",
        namespace: str = "",
        batch_size: int = 100,
    ) -> None:
        try:
            total = len(documents)
            for i in range(0, total, batch_size):
                batch = documents[i : i + batch_size]
                embeddings = self.get_embeddings_batch([d[text_field] for d in batch])
                vectors = [
                    (d[id_field], emb, {k: v for k, v in d.items() if k != id_field})
                    for d, emb in zip(batch, embeddings)
                ]
                self.index.upsert(vectors=vectors, namespace=namespace)
                logger.debug("Обработано {}/{} документов", min(i + batch_size, total), total)
            logger.info("Все {} документов записаны в namespace '{}'", total, namespace)
        except Exception as exc:
            raise Exception(f"Ошибка при записи документов: {exc}") from exc

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query_by_vector(
        self,
        vector: List[float],
        top_k: int = 10,
        namespace: str = "",
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> Dict[str, Any]:
        try:
            return self.index.query(
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                filter=filter_dict,
                include_metadata=include_metadata,
                include_values=include_values,
            )
        except Exception as exc:
            raise Exception(f"Ошибка при поиске по вектору: {exc}") from exc

    def query_by_text(
        self,
        text: str,
        top_k: int = 10,
        namespace: str = "",
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> Dict[str, Any]:
        try:
            return self.query_by_vector(
                vector=self.get_embedding(text),
                top_k=top_k,
                namespace=namespace,
                filter_dict=filter_dict,
                include_metadata=include_metadata,
                include_values=include_values,
            )
        except Exception as exc:
            raise Exception(f"Ошибка при поиске по тексту: {exc}") from exc

    def query_by_id(
        self,
        doc_id: str,
        top_k: int = 10,
        namespace: str = "",
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        try:
            return self.index.query(
                id=doc_id,
                top_k=top_k,
                namespace=namespace,
                filter=filter_dict,
                include_metadata=include_metadata,
            )
        except Exception as exc:
            raise Exception(f"Ошибка при поиске по ID: {exc}") from exc

    def fetch(self, ids: List[str], namespace: str = "") -> Dict[str, Any]:
        try:
            return self.index.fetch(ids=ids, namespace=namespace)
        except Exception as exc:
            raise Exception(f"Ошибка при получении векторов: {exc}") from exc

    def get_stats(self) -> Dict[str, Any]:
        try:
            return self.index.describe_index_stats()
        except Exception as exc:
            raise Exception(f"Ошибка при получении статистики: {exc}") from exc

    def list_namespaces(self) -> List[str]:
        return list(self.get_stats().get("namespaces", {}).keys())

    # ------------------------------------------------------------------
    # Delete / Update
    # ------------------------------------------------------------------

    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: bool = False,
        namespace: str = "",
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            if delete_all:
                self.index.delete(delete_all=True, namespace=namespace)
                logger.info("Все векторы удалены из namespace '{}'", namespace)
            elif ids:
                self.index.delete(ids=ids, namespace=namespace)
                logger.info("Удалено {} векторов из namespace '{}'", len(ids), namespace)
            elif filter_dict:
                self.index.delete(filter=filter_dict, namespace=namespace)
            else:
                logger.warning("delete() вызван без параметров: укажи ids, delete_all=True или filter_dict")
        except Exception as exc:
            raise Exception(f"Ошибка при удалении векторов: {exc}") from exc

    def update_metadata(self, doc_id: str, metadata: Dict[str, Any], namespace: str = "") -> None:
        try:
            self.index.update(id=doc_id, set_metadata=metadata, namespace=namespace)
            logger.debug("Метаданные документа '{}' обновлены", doc_id)
        except Exception as exc:
            raise Exception(f"Ошибка при обновлении метаданных: {exc}") from exc

    # ------------------------------------------------------------------
    # Smart upsert (with duplicate check)
    # ------------------------------------------------------------------

    def check_similarity(
        self,
        text: str,
        namespace: str = "",
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        threshold_high: Optional[float] = None,
        threshold_low: Optional[float] = None,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Возвращает ('duplicate'|'similar'|'new', best_match)."""
        high = threshold_high if threshold_high is not None else SIMILARITY_THRESHOLD_HIGH
        low = threshold_low if threshold_low is not None else SIMILARITY_THRESHOLD_LOW
        try:
            results = self.query_by_vector(
                vector=self.get_embedding(text),
                top_k=top_k,
                namespace=namespace,
                filter_dict=filter_dict,
                include_metadata=True,
            )
            if not results.get("matches"):
                return "new", None
            best = results["matches"][0]
            score = best.get("score", 0.0)
            if score >= high:
                return "duplicate", best
            if score >= low:
                return "similar", best
            return "new", best
        except Exception as exc:
            raise Exception(f"Ошибка при проверке сходства: {exc}") from exc

    def smart_upsert_text(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: str = "",
        check_duplicates: bool = True,
        update_if_duplicate: bool = True,
        threshold_high: Optional[float] = None,
        threshold_low: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Записывает текст с проверкой косинусного сходства."""
        result: Dict[str, Any] = {
            "action": None,
            "doc_id": doc_id,
            "similarity_status": None,
            "similarity_score": None,
            "matched_id": None,
        }
        try:
            if check_duplicates:
                status, match = self.check_similarity(
                    text=text,
                    namespace=namespace,
                    filter_dict=filter_dict,
                    threshold_high=threshold_high,
                    threshold_low=threshold_low,
                )
                result["similarity_status"] = status
                if match:
                    result["similarity_score"] = match.get("score")
                    result["matched_id"] = match.get("id")

                if status == "duplicate":
                    if update_if_duplicate and match:
                        mid = match["id"]
                        meta = dict(metadata or {})
                        meta["text"] = text
                        self.update_metadata(mid, meta, namespace)
                        result.update({"action": "updated", "doc_id": mid})
                        logger.debug("duplicate (score: {:.4f}) — обновлена запись '{}'", match["score"], mid)
                    else:
                        result["action"] = "skipped"
                        logger.debug("duplicate (score: {:.4f}) — запись пропущена", match.get("score", 0))
                    return result

                if status == "similar":
                    logger.debug("similar (score: {:.4f}) — записываем как новую", match.get("score", 0))
                else:
                    logger.debug("new — записываем в базу")

            self.upsert_text(text=text, doc_id=doc_id, metadata=metadata, namespace=namespace)
            result["action"] = "created"
            return result
        except Exception as exc:
            raise Exception(f"Ошибка при умной записи текста: {exc}") from exc

    def smart_upsert_documents(
        self,
        documents: List[Dict[str, Any]],
        id_field: str = "id",
        text_field: str = "text",
        namespace: str = "",
        check_duplicates: bool = True,
        update_if_duplicate: bool = True,
        threshold_high: Optional[float] = None,
        threshold_low: Optional[float] = None,
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """Пакетная запись документов с опциональной проверкой дубликатов."""
        stats: Dict[str, Any] = {
            "total": len(documents),
            "created": 0,
            "updated": 0,
            "skipped": 0,
            "results": [],
        }
        try:
            if check_duplicates:
                for i, doc in enumerate(documents, 1):
                    print(f"\nОбработка документа {i}/{len(documents)}: {doc[id_field]}")
                    res = self.smart_upsert_text(
                        text=doc[text_field],
                        doc_id=doc[id_field],
                        metadata={k: v for k, v in doc.items() if k != id_field},
                        namespace=namespace,
                        check_duplicates=True,
                        update_if_duplicate=update_if_duplicate,
                        threshold_high=threshold_high,
                        threshold_low=threshold_low,
                    )
                    stats["results"].append(res)
                    stats[res["action"]] = stats.get(res["action"], 0) + 1
            else:
                self.upsert_documents(
                    documents=documents,
                    id_field=id_field,
                    text_field=text_field,
                    namespace=namespace,
                    batch_size=batch_size,
                )
                stats["created"] = len(documents)

            logger.info(
                "Итог: всего {}, создано {}, обновлено {}, пропущено {}",
                stats["total"], stats["created"], stats["updated"], stats["skipped"],
            )
            return stats
        except Exception as exc:
            raise Exception(f"Ошибка при умной записи документов: {exc}") from exc
