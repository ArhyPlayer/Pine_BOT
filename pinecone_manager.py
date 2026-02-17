"""
Модуль для управления векторной базой данных Pinecone.

Этот модуль предоставляет класс PineconeManager для работы с Pinecone,
включая запись и чтение векторов, работу с документами и текстовый поиск.
"""

import os
from typing import List, Dict, Any, Optional, Union, Tuple
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv


# Глобальные настройки для определения дубликатов
# Порог косинусного сходства (значения от 0.0 до 1.0):
# - 1.0 = идентичные векторы
# - 0.9-0.99 = очень похожие (скорее всего дубликаты)
# - 0.7-0.9 = похожие (могут быть вариации одной темы)
# - < 0.7 = разная информация
SIMILARITY_THRESHOLD_HIGH = 0.80  # Выше этого значения считается дубликатом
SIMILARITY_THRESHOLD_LOW = 0.75    # Ниже этого значения - новая информация


class PineconeManager:
    """
    Класс для управления операциями с векторной базой данных Pinecone.
    
    Поддерживает:
    - Запись векторов и документов
    - Поиск по векторам и тексту
    - Управление метаданными
    - Удаление записей
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        dimension: int = 1536
    ):
        """
        Инициализирует менеджер Pinecone.
        
        Args:
            api_key: API ключ Pinecone (если None, берется из .env)
            index_name: Имя индекса Pinecone (если None, берется из .env)
            embedding_model: Модель для эмбеддингов (если None, берется из .env)
            openai_api_key: API ключ OpenAI (если None, берется из .env)
            openai_base_url: Base URL для OpenAI API (если None, берется из .env)
            dimension: Размерность векторов (по умолчанию 1536 для text-embedding-3-small)
        """
        # Загрузка переменных окружения
        load_dotenv()
        
        # Инициализация параметров
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.dimension = dimension
        
        if not self.api_key:
            raise ValueError("Необходимо указать PINECONE_API_KEY в .env или при инициализации")
        
        if not self.index_name:
            raise ValueError("Необходимо указать PINECONE_INDEX_NAME в .env или при инициализации")
        
        # Инициализация Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        
        # Инициализация OpenAI для работы с эмбеддингами
        openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        openai_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
        
        if not openai_key:
            raise ValueError("Необходимо указать OPENAI_API_KEY в .env или при инициализации")
        
        # Создание клиента OpenAI
        if openai_url:
            self.openai_client = OpenAI(api_key=openai_key, base_url=openai_url)
        else:
            self.openai_client = OpenAI(api_key=openai_key)
        
        # Подключение к индексу
        self.index = None
        self._connect_to_index()
    
    def _connect_to_index(self):
        """Подключается к существующему индексу или создает новый."""
        try:
            # Проверяем, существует ли индекс
            if self.index_name not in [idx.name for idx in self.pc.list_indexes()]:
                print(f"Индекс '{self.index_name}' не найден. Создание нового индекса...")
                self.create_index()
            
            # Подключаемся к индексу
            self.index = self.pc.Index(self.index_name)
            print(f"Успешно подключено к индексу '{self.index_name}'")
        except Exception as e:
            raise Exception(f"Ошибка при подключении к индексу: {str(e)}")
    
    def create_index(
        self,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1"
    ):
        """
        Создает новый индекс в Pinecone.
        
        Args:
            dimension: Размерность векторов (если None, используется self.dimension)
            metric: Метрика для сравнения векторов (cosine, euclidean, dotproduct)
            cloud: Облачный провайдер (aws, gcp, azure)
            region: Регион для размещения индекса
        """
        dim = dimension or self.dimension
        
        try:
            self.pc.create_index(
                name=self.index_name,
                dimension=dim,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )
            print(f"Индекс '{self.index_name}' успешно создан с размерностью {dim}")
        except Exception as e:
            raise Exception(f"Ошибка при создании индекса: {str(e)}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Получает эмбеддинг для текста с помощью OpenAI.
        
        Args:
            text: Текст для векторизации
            
        Returns:
            Вектор-эмбеддинг
        """
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Ошибка при получении эмбеддинга: {str(e)}")
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Получает эмбеддинги для списка текстов.
        
        Args:
            texts: Список текстов для векторизации
            
        Returns:
            Список векторов-эмбеддингов
        """
        try:
            response = self.openai_client.embeddings.create(
                input=texts,
                model=self.embedding_model
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise Exception(f"Ошибка при получении эмбеддингов: {str(e)}")
    
    def upsert_vectors(
        self,
        vectors: List[tuple],
        namespace: str = ""
    ):
        """
        Записывает векторы напрямую в Pinecone.
        
        Args:
            vectors: Список кортежей (id, vector, metadata)
                     или (id, vector) без метаданных
            namespace: Пространство имен для векторов
            
        Example:
            vectors = [
                ("id1", [0.1, 0.2, ...], {"text": "пример", "category": "test"}),
                ("id2", [0.3, 0.4, ...], {"text": "второй пример"})
            ]
        """
        try:
            self.index.upsert(vectors=vectors, namespace=namespace)
            print(f"Записано {len(vectors)} векторов в namespace '{namespace}'")
        except Exception as e:
            raise Exception(f"Ошибка при записи векторов: {str(e)}")
    
    def upsert_documents(
        self,
        documents: List[Dict[str, Any]],
        id_field: str = "id",
        text_field: str = "text",
        namespace: str = "",
        batch_size: int = 100
    ):
        """
        Записывает документы с автоматической векторизацией.
        
        Args:
            documents: Список документов (словарей с данными)
            id_field: Название поля с ID документа
            text_field: Название поля с текстом для векторизации
            namespace: Пространство имен
            batch_size: Размер пакета для обработки
            
        Example:
            documents = [
                {"id": "doc1", "text": "Текст документа", "category": "tech"},
                {"id": "doc2", "text": "Другой документ", "author": "John"}
            ]
        """
        try:
            total = len(documents)
            for i in range(0, total, batch_size):
                batch = documents[i:i + batch_size]
                
                # Извлекаем тексты для векторизации
                texts = [doc[text_field] for doc in batch]
                
                # Получаем эмбеддинги
                embeddings = self.get_embeddings_batch(texts)
                
                # Подготавливаем векторы для загрузки
                vectors = []
                for doc, embedding in zip(batch, embeddings):
                    doc_id = doc[id_field]
                    # Все поля документа сохраняем как метаданные
                    metadata = {k: v for k, v in doc.items() if k != id_field}
                    vectors.append((doc_id, embedding, metadata))
                
                # Загружаем в Pinecone
                self.index.upsert(vectors=vectors, namespace=namespace)
                
                print(f"Обработано {min(i + batch_size, total)}/{total} документов")
            
            print(f"Все {total} документов успешно записаны в namespace '{namespace}'")
        except Exception as e:
            raise Exception(f"Ошибка при записи документов: {str(e)}")
    
    def upsert_text(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: str = ""
    ):
        """
        Записывает один текстовый документ с векторизацией.
        
        Args:
            text: Текст для записи
            doc_id: ID документа
            metadata: Дополнительные метаданные
            namespace: Пространство имен
        """
        try:
            # Получаем эмбеддинг
            embedding = self.get_embedding(text)
            
            # Подготавливаем метаданные
            meta = metadata or {}
            meta["text"] = text
            
            # Записываем
            self.index.upsert(
                vectors=[(doc_id, embedding, meta)],
                namespace=namespace
            )
            print(f"Документ '{doc_id}' успешно записан")
        except Exception as e:
            raise Exception(f"Ошибка при записи текста: {str(e)}")
    
    def query_by_vector(
        self,
        vector: List[float],
        top_k: int = 10,
        namespace: str = "",
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_values: bool = False
    ) -> Dict[str, Any]:
        """
        Поиск похожих векторов по заданному вектору.
        
        Args:
            vector: Вектор для поиска
            top_k: Количество результатов
            namespace: Пространство имен
            filter_dict: Фильтр метаданных (например, {"category": "tech"})
            include_metadata: Включить метаданные в результаты
            include_values: Включить значения векторов
            
        Returns:
            Результаты поиска
        """
        try:
            results = self.index.query(
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                filter=filter_dict,
                include_metadata=include_metadata,
                include_values=include_values
            )
            return results
        except Exception as e:
            raise Exception(f"Ошибка при поиске по вектору: {str(e)}")
    
    def query_by_text(
        self,
        text: str,
        top_k: int = 10,
        namespace: str = "",
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_values: bool = False
    ) -> Dict[str, Any]:
        """
        Поиск похожих документов по текстовому запросу.
        
        Args:
            text: Текстовый запрос
            top_k: Количество результатов
            namespace: Пространство имен
            filter_dict: Фильтр метаданных
            include_metadata: Включить метаданные в результаты
            include_values: Включить значения векторов
            
        Returns:
            Результаты поиска
        """
        try:
            # Векторизуем текст запроса
            query_vector = self.get_embedding(text)
            
            # Выполняем поиск
            results = self.query_by_vector(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                filter_dict=filter_dict,
                include_metadata=include_metadata,
                include_values=include_values
            )
            return results
        except Exception as e:
            raise Exception(f"Ошибка при поиске по тексту: {str(e)}")
    
    def query_by_id(
        self,
        doc_id: str,
        top_k: int = 10,
        namespace: str = "",
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Поиск похожих документов по ID существующего документа.
        
        Args:
            doc_id: ID документа для использования в качестве запроса
            top_k: Количество результатов
            namespace: Пространство имен
            filter_dict: Фильтр метаданных
            include_metadata: Включить метаданные в результаты
            
        Returns:
            Результаты поиска
        """
        try:
            results = self.index.query(
                id=doc_id,
                top_k=top_k,
                namespace=namespace,
                filter=filter_dict,
                include_metadata=include_metadata
            )
            return results
        except Exception as e:
            raise Exception(f"Ошибка при поиске по ID: {str(e)}")
    
    def fetch(
        self,
        ids: List[str],
        namespace: str = ""
    ) -> Dict[str, Any]:
        """
        Получает конкретные векторы по их ID.
        
        Args:
            ids: Список ID для получения
            namespace: Пространство имен
            
        Returns:
            Словарь с векторами
        """
        try:
            results = self.index.fetch(ids=ids, namespace=namespace)
            return results
        except Exception as e:
            raise Exception(f"Ошибка при получении векторов: {str(e)}")
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: bool = False,
        namespace: str = "",
        filter_dict: Optional[Dict[str, Any]] = None
    ):
        """
        Удаляет векторы из индекса.
        
        Args:
            ids: Список ID для удаления
            delete_all: Удалить все векторы в namespace
            namespace: Пространство имен
            filter_dict: Фильтр для удаления (например, {"category": "old"})
        """
        try:
            if delete_all:
                self.index.delete(delete_all=True, namespace=namespace)
                print(f"Все векторы удалены из namespace '{namespace}'")
            elif ids:
                self.index.delete(ids=ids, namespace=namespace)
                print(f"Удалено {len(ids)} векторов из namespace '{namespace}'")
            elif filter_dict:
                self.index.delete(filter=filter_dict, namespace=namespace)
                print(f"Удалены векторы с фильтром {filter_dict} из namespace '{namespace}'")
            else:
                print("Необходимо указать ids, delete_all=True или filter_dict")
        except Exception as e:
            raise Exception(f"Ошибка при удалении векторов: {str(e)}")
    
    def update_metadata(
        self,
        doc_id: str,
        metadata: Dict[str, Any],
        namespace: str = ""
    ):
        """
        Обновляет метаданные существующего вектора.
        
        Args:
            doc_id: ID документа
            metadata: Новые метаданные
            namespace: Пространство имен
        """
        try:
            self.index.update(
                id=doc_id,
                set_metadata=metadata,
                namespace=namespace
            )
            print(f"Метаданные документа '{doc_id}' обновлены")
        except Exception as e:
            raise Exception(f"Ошибка при обновлении метаданных: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получает статистику индекса.
        
        Returns:
            Статистика индекса (количество векторов, размерность и т.д.)
        """
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            raise Exception(f"Ошибка при получении статистики: {str(e)}")
    
    def list_namespaces(self) -> List[str]:
        """
        Получает список всех пространств имен в индексе.
        
        Returns:
            Список названий пространств имен
        """
        try:
            stats = self.get_stats()
            return list(stats.get('namespaces', {}).keys())
        except Exception as e:
            raise Exception(f"Ошибка при получении списка namespaces: {str(e)}")
    
    def check_similarity(
        self,
        text: str,
        namespace: str = "",
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        threshold_high: Optional[float] = None,
        threshold_low: Optional[float] = None
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Проверяет сходство текста с существующими записями в базе.
        
        Args:
            text: Текст для проверки
            namespace: Пространство имен
            top_k: Количество похожих записей для проверки
            filter_dict: Фильтр метаданных
            threshold_high: Порог высокого сходства (дубликат)
            threshold_low: Порог низкого сходства (новая информация)
            
        Returns:
            Tuple[status, match]:
                - status: "duplicate" (дубликат), "similar" (похожий), "new" (новый)
                - match: Данные наиболее похожей записи (если найдена) или None
        """
        # Используем глобальные пороги, если не указаны
        high_threshold = threshold_high if threshold_high is not None else SIMILARITY_THRESHOLD_HIGH
        low_threshold = threshold_low if threshold_low is not None else SIMILARITY_THRESHOLD_LOW
        
        try:
            # Получаем эмбеддинг текста
            query_vector = self.get_embedding(text)
            
            # Ищем похожие записи
            results = self.query_by_vector(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                filter_dict=filter_dict,
                include_metadata=True
            )
            
            # Проверяем, есть ли совпадения
            if not results.get('matches'):
                return "new", None
            
            # Берем самое похожее совпадение
            best_match = results['matches'][0]
            similarity_score = best_match.get('score', 0.0)
            
            # Определяем статус на основе порога
            if similarity_score >= high_threshold:
                return "duplicate", best_match
            elif similarity_score >= low_threshold:
                return "similar", best_match
            else:
                return "new", best_match
                
        except Exception as e:
            raise Exception(f"Ошибка при проверке сходства: {str(e)}")
    
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
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Умная запись текста с проверкой дубликатов.
        
        Алгоритм:
        1. Проверяет косинусное сходство с существующими записями
        2. Если сходство высокое (дубликат) - обновляет существующую запись или пропускает
        3. Если сходство низкое (новая информация) - записывает новую запись
        
        Args:
            text: Текст для записи
            doc_id: ID документа
            metadata: Дополнительные метаданные
            namespace: Пространство имен
            check_duplicates: Проверять дубликаты перед записью
            update_if_duplicate: Обновлять запись при обнаружении дубликата
            threshold_high: Порог высокого сходства
            threshold_low: Порог низкого сходства
            filter_dict: Фильтр для поиска похожих записей
            
        Returns:
            Словарь с результатом операции:
            {
                "action": "created" | "updated" | "skipped",
                "doc_id": str,
                "similarity_status": "new" | "similar" | "duplicate",
                "similarity_score": float (если найдено совпадение),
                "matched_id": str (если найдено совпадение)
            }
        """
        result = {
            "action": None,
            "doc_id": doc_id,
            "similarity_status": None,
            "similarity_score": None,
            "matched_id": None
        }
        
        try:
            # Если проверка дубликатов включена
            if check_duplicates:
                status, match = self.check_similarity(
                    text=text,
                    namespace=namespace,
                    filter_dict=filter_dict,
                    threshold_high=threshold_high,
                    threshold_low=threshold_low
                )
                
                result["similarity_status"] = status
                
                if match:
                    result["similarity_score"] = match.get('score')
                    result["matched_id"] = match.get('id')
                
                # Обработка на основе статуса сходства
                if status == "duplicate":
                    if update_if_duplicate and match:
                        # Обновляем существующую запись
                        matched_id = match.get('id')
                        meta = metadata or {}
                        meta["text"] = text
                        self.update_metadata(
                            doc_id=matched_id,
                            metadata=meta,
                            namespace=namespace
                        )
                        result["action"] = "updated"
                        result["doc_id"] = matched_id
                        print(f"Обнаружен дубликат (score: {match.get('score'):.4f}). Обновлена запись '{matched_id}'")
                    else:
                        # Пропускаем запись
                        result["action"] = "skipped"
                        print(f"Обнаружен дубликат (score: {match.get('score'):.4f}). Запись пропущена.")
                    return result
                
                elif status == "similar":
                    # Похожая информация, но не дубликат - записываем как новую
                    print(f"Найдена похожая запись (score: {match.get('score'):.4f}), но записываем как новую.")
                
                else:  # status == "new"
                    print("Новая уникальная информация. Записываем в базу.")
            
            # Записываем новую запись
            self.upsert_text(
                text=text,
                doc_id=doc_id,
                metadata=metadata,
                namespace=namespace
            )
            result["action"] = "created"
            
            return result
            
        except Exception as e:
            raise Exception(f"Ошибка при умной записи текста: {str(e)}")
    
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
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Умная запись документов с проверкой дубликатов.
        
        Args:
            documents: Список документов
            id_field: Название поля с ID
            text_field: Название поля с текстом
            namespace: Пространство имен
            check_duplicates: Проверять дубликаты
            update_if_duplicate: Обновлять при дубликатах
            threshold_high: Порог высокого сходства
            threshold_low: Порог низкого сходства
            batch_size: Размер пакета (не используется при проверке дубликатов)
            
        Returns:
            Словарь со статистикой:
            {
                "total": int,
                "created": int,
                "updated": int,
                "skipped": int,
                "results": List[Dict]
            }
        """
        stats = {
            "total": len(documents),
            "created": 0,
            "updated": 0,
            "skipped": 0,
            "results": []
        }
        
        try:
            if check_duplicates:
                # При проверке дубликатов обрабатываем документы по одному
                for i, doc in enumerate(documents, 1):
                    text = doc[text_field]
                    doc_id = doc[id_field]
                    metadata = {k: v for k, v in doc.items() if k != id_field}
                    
                    print(f"\nОбработка документа {i}/{len(documents)}: {doc_id}")
                    
                    result = self.smart_upsert_text(
                        text=text,
                        doc_id=doc_id,
                        metadata=metadata,
                        namespace=namespace,
                        check_duplicates=True,
                        update_if_duplicate=update_if_duplicate,
                        threshold_high=threshold_high,
                        threshold_low=threshold_low
                    )
                    
                    stats["results"].append(result)
                    
                    if result["action"] == "created":
                        stats["created"] += 1
                    elif result["action"] == "updated":
                        stats["updated"] += 1
                    elif result["action"] == "skipped":
                        stats["skipped"] += 1
            else:
                # Без проверки дубликатов - стандартная пакетная запись
                self.upsert_documents(
                    documents=documents,
                    id_field=id_field,
                    text_field=text_field,
                    namespace=namespace,
                    batch_size=batch_size
                )
                stats["created"] = len(documents)
            
            print(f"\n=== Итоговая статистика ===")
            print(f"Всего документов: {stats['total']}")
            print(f"Создано новых: {stats['created']}")
            print(f"Обновлено: {stats['updated']}")
            print(f"Пропущено: {stats['skipped']}")
            
            return stats
            
        except Exception as e:
            raise Exception(f"Ошибка при умной записи документов: {str(e)}")


# Пример использования
if __name__ == "__main__":
    # Инициализация менеджера
    pm = PineconeManager()
    
    print("=== Пример 1: Умная запись с проверкой дубликатов ===")
    
    # Первая запись
    result1 = pm.smart_upsert_text(
        text="Пользователь любит пиццу с грибами и сыром",
        doc_id="memory_1",
        metadata={"user_id": "user123", "category": "preferences"}
    )
    print(f"Результат 1: {result1}")
    
    # Попытка записать похожую информацию (должна быть обнаружена как дубликат)
    result2 = pm.smart_upsert_text(
        text="Пользователь предпочитает пиццу с грибами",
        doc_id="memory_2",
        metadata={"user_id": "user123", "category": "preferences"}
    )
    print(f"Результат 2: {result2}")
    
    # Запись новой информации
    result3 = pm.smart_upsert_text(
        text="Пользователь занимается программированием на Python",
        doc_id="memory_3",
        metadata={"user_id": "user123", "category": "skills"}
    )
    print(f"Результат 3: {result3}")
    
    print("\n=== Пример 2: Умная запись нескольких документов ===")
    
    documents = [
        {
            "id": "doc1",
            "text": "Искусственный интеллект - это область компьютерных наук",
            "category": "tech"
        },
        {
            "id": "doc2",
            "text": "Машинное обучение - это подмножество искусственного интеллекта",
            "category": "tech"
        },
        {
            "id": "doc3",
            "text": "ИИ является частью компьютерных наук",  # Дубликат doc1
            "category": "tech"
        }
    ]
    
    stats = pm.smart_upsert_documents(
        documents=documents,
        check_duplicates=True,
        update_if_duplicate=False  # Пропускать дубликаты
    )
    
    print("\n=== Пример 3: Проверка сходства вручную ===")
    
    status, match = pm.check_similarity(
        text="Что такое машинное обучение?",
        top_k=3
    )
    
    print(f"Статус сходства: {status}")
    if match:
        print(f"Похожая запись: {match['id']}")
        print(f"Оценка сходства: {match['score']:.4f}")
        print(f"Текст: {match.get('metadata', {}).get('text', 'N/A')}")
    
    print("\n=== Пример 4: Поиск по тексту ===")
    
    results = pm.query_by_text(
        text="Любимая еда пользователя",
        top_k=3
    )
    
    print("\nРезультаты поиска:")
    for match in results['matches']:
        print(f"ID: {match['id']}, Score: {match['score']:.4f}")
        if 'metadata' in match:
            print(f"Текст: {match['metadata'].get('text', 'N/A')}")
    
    print("\n=== Пример 5: Статистика индекса ===")
    
    stats = pm.get_stats()
    print(f"Всего векторов: {stats.get('total_vector_count', 0)}")
    print(f"Размерность: {stats.get('dimension', 0)}")
    
    print("\n=== Пример 6: Настройка порогов сходства ===")
    
    # Более строгая проверка дубликатов (только очень похожие считаются дубликатами)
    result_strict = pm.smart_upsert_text(
        text="Пользователь иногда ест пиццу",
        doc_id="memory_4",
        metadata={"user_id": "user123"},
        threshold_high=0.95,  # Очень строгий порог
        threshold_low=0.85
    )
    print(f"Результат со строгими порогами: {result_strict}")
