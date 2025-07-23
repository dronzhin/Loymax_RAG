from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging
import pickle
import os
from typing import List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# Глобальная переменная для модели
_MODEL: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """Получает или создает экземпляр модели."""
    global _MODEL
    if _MODEL is None:
        logger.info("Загрузка модели SentenceTransformer...")
        _MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _MODEL


def vectorize_text(texts: List[str]) -> np.ndarray:
    """
    Векторизует тексты с использованием SentenceTransformer.

    Args:
        texts: Список текстов для векторизации

    Returns:
        np.ndarray: Массив эмбеддингов
    """
    try:
        model = get_model()
        logger.info(f"Кодирование {len(texts)} текстов...")
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        logger.info(f"Успешно кодировано {len(texts)} текстов.")
        return embeddings
    except Exception as e:
        logger.error(f"Ошибка кодирования текстов: {str(e)}", exc_info=True)
        raise


def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Создает индекс FAISS из эмбеддингов.

    Args:
        embeddings: Массив эмбеддингов

    Returns:
        faiss.Index: Созданный индекс FAISS
    """
    try:
        if embeddings.size == 0:
            raise ValueError("Получен пустой массив эмбеддингов")

        dimension = embeddings.shape[1]
        logger.info(f"Создание индекса FAISS с размерностью {dimension}")

        # Используем IndexFlatIP для поиска по косинусному сходству
        index = faiss.IndexFlatIP(dimension)

        # Добавляем нормализацию для косинусного сходства
        faiss.normalize_L2(embeddings)

        index.add(embeddings)
        logger.info(f"Создан индекс FAISS с {index.ntotal} векторами")
        return index
    except Exception as e:
        logger.error(f"Ошибка создания индекса FAISS: {str(e)}", exc_info=True)
        raise


def save_faiss_index_and_metadata(
        index: faiss.Index,
        texts: List[str],
        index_path: str,
        metadata_path: str
) -> None:
    """
    Сохраняет индекс FAISS и связанные метаданные.

    Args:
        index: Индекс FAISS для сохранения
        texts: Список текстовых метаданных
        index_path: Путь для сохранения индекса
        metadata_path: Путь для сохранения метаданных
    """
    try:
        # Создаем директории при необходимости
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        # Проверка соответствия размеров
        if len(texts) != index.ntotal:
            raise ValueError("Количество текстов не соответствует количеству векторов в индексе")

        logger.info(f"Сохранение индекса в {index_path}")
        faiss.write_index(index, index_path)

        logger.info(f"Сохранение метаданных в {metadata_path}")
        with open(metadata_path, 'wb') as f:
            pickle.dump(texts, f)

        logger.info("Индекс и метаданные успешно сохранены")
    except Exception as e:
        logger.error(f"Ошибка сохранения индекса и метаданных: {str(e)}", exc_info=True)
        raise