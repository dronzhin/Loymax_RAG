from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging

logger = logging.getLogger(__name__)

def vectorize_text(texts: list[str]) -> np.ndarray:
    """Векторизует тексты с использованием SentenceTransformer."""
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    try:
        embeddings = model.encode(texts)
        logger.info(f"Успешно кодировали {len(texts)} текстов.")
        return embeddings
    except Exception as e:
        logger.error(f"Не удалось кодировать тексты: {e}")
        raise

def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Создает индекс FAISS из эмбеддингов."""
    dimension = embeddings.shape[1]
    try:
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        logger.info(f"Успешно создан индекс FAISS с {len(embeddings)} векторами.")
        return index
    except Exception as e:
        logger.error(f"Не удалось создать индекс FAISS: {e}")
        raise