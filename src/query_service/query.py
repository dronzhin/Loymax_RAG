import faiss
from sentence_transformers import SentenceTransformer
import logging
import requests

# Создаем логгер для текущего модуля
logger = logging.getLogger(__name__)

def load_faiss_index(index_path: str) -> faiss.Index:
    """Загружает индекс FAISS из файла."""
    try:
        index = faiss.read_index(index_path)
        logger.info(f"Индекс FAISS загружен из {index_path}")
        return index
    except FileNotFoundError:
        logger.error(f"Файл не найден: {index_path}")
        raise

def query_index(index: faiss.Index, query_text: str, k: int = 5) -> list[str]:
    """Запрашивает индекс FAISS и возвращает ближайшие соседи."""
    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        query_embedding = model.encode([query_text])
        _, indices = index.search(query_embedding, k)
        logger.info(f"Найдено {len(indices[0])} ближайших соседей для запроса: '{query_text}'")
        return indices[0]
    except Exception as e:
        logger.error(f"Ошибка при запросе индекса FAISS: {e}")
        raise

def query_ollama(prompt: str, model: str = "mistral") -> str:
    """Запрашивает Ollama для генерации ответа на основе промпта."""
    url = "http://localhost:8985/api/generate"
    payload = {
        "model": model,
        "prompt": prompt
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "")
    else:
        raise Exception(f"Error querying Ollama: {response.status_code}")

def generate_answer(question: str, context: str) -> str:
    """Генерирует ответ на вопрос на основе контекста."""
    prompt = f"Контекст: {context}\nВопрос: {question}\nОтвет:"
    return query_ollama(prompt)