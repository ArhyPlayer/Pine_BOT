"""
Примеры использования PineconeManager.

Запускать из корня проекта:
    python examples/pinecone_usage.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from memory.store import PineconeManager

pm = PineconeManager()

# ------------------------------------------------------------------
# Пример 1: Умная запись с проверкой дубликатов
# ------------------------------------------------------------------
print("=== Пример 1: Умная запись с проверкой дубликатов ===")

result1 = pm.smart_upsert_text(
    text="Пользователь любит пиццу с грибами и сыром",
    doc_id="memory_1",
    metadata={"user_id": "user123", "category": "preferences"},
)
print(f"Результат 1: {result1}")

# Похожая информация — должна быть обнаружена как дубликат
result2 = pm.smart_upsert_text(
    text="Пользователь предпочитает пиццу с грибами",
    doc_id="memory_2",
    metadata={"user_id": "user123", "category": "preferences"},
)
print(f"Результат 2: {result2}")

# Новая информация
result3 = pm.smart_upsert_text(
    text="Пользователь занимается программированием на Python",
    doc_id="memory_3",
    metadata={"user_id": "user123", "category": "skills"},
)
print(f"Результат 3: {result3}")

# ------------------------------------------------------------------
# Пример 2: Умная запись нескольких документов
# ------------------------------------------------------------------
print("\n=== Пример 2: Умная запись нескольких документов ===")

documents = [
    {
        "id": "doc1",
        "text": "Искусственный интеллект — это область компьютерных наук",
        "category": "tech",
    },
    {
        "id": "doc2",
        "text": "Машинное обучение — это подмножество искусственного интеллекта",
        "category": "tech",
    },
    {
        "id": "doc3",
        "text": "ИИ является частью компьютерных наук",  # дубликат doc1
        "category": "tech",
    },
]

stats = pm.smart_upsert_documents(
    documents=documents,
    check_duplicates=True,
    update_if_duplicate=False,  # пропускать дубликаты
)

# ------------------------------------------------------------------
# Пример 3: Проверка сходства вручную
# ------------------------------------------------------------------
print("\n=== Пример 3: Проверка сходства вручную ===")

status, match = pm.check_similarity(
    text="Что такое машинное обучение?",
    top_k=3,
)
print(f"Статус сходства: {status}")
if match:
    print(f"Похожая запись: {match['id']}")
    print(f"Оценка сходства: {match['score']:.4f}")
    print(f"Текст: {match.get('metadata', {}).get('text', 'N/A')}")

# ------------------------------------------------------------------
# Пример 4: Поиск по тексту
# ------------------------------------------------------------------
print("\n=== Пример 4: Поиск по тексту ===")

results = pm.query_by_text(
    text="Любимая еда пользователя",
    top_k=3,
)
print("Результаты поиска:")
for m in results.get("matches", []):
    print(f"  ID: {m['id']}, Score: {m['score']:.4f}")
    print(f"  Текст: {m.get('metadata', {}).get('text', 'N/A')}")

# ------------------------------------------------------------------
# Пример 5: Статистика индекса
# ------------------------------------------------------------------
print("\n=== Пример 5: Статистика индекса ===")

index_stats = pm.get_stats()
print(f"Всего векторов: {index_stats.get('total_vector_count', 0)}")
print(f"Размерность: {index_stats.get('dimension', 0)}")

# ------------------------------------------------------------------
# Пример 6: Настройка порогов сходства
# ------------------------------------------------------------------
print("\n=== Пример 6: Настройка порогов сходства ===")

result_strict = pm.smart_upsert_text(
    text="Пользователь иногда ест пиццу",
    doc_id="memory_4",
    metadata={"user_id": "user123"},
    threshold_high=0.95,  # очень строгий порог
    threshold_low=0.85,
)
print(f"Результат со строгими порогами: {result_strict}")
