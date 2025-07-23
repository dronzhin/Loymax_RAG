import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

# Создаем логгер для текущего модуля
logger = logging.getLogger(__name__)

def load_faiss_index(index_path: str) -> faiss.Index:
    """Загружает индекс FAISS из файла."""
    try:
        index = faiss.read_index(index_path)
        logger.info(f"FAISS index loaded from {index_path}")
        return index
    except FileNotFoundError:
        logger.error(f"File not found: {index_path}")
        raise

def query_index(index: faiss.Index, query_text: str, k: int = 5) -> List[str]:
    """Запрашивает индекс FAISS и возвращает ближайшие соседи."""
    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        query_embedding = model.encode([query_text])
        distances, indices = index.search(query_embedding, k)
        logger.info(f"Found {len(indices[0])} nearest neighbors for query: '{query_text}'")
        return indices[0]
    except Exception as e:
        logger.error(f"Error querying FAISS index: {e}")
        raise