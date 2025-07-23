import faiss
from sentence_transformers import SentenceTransformer
import logging
import requests
import json
import os
from typing import List, Tuple, Optional
import pickle

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'app.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
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


def load_faiss_index(index_path: str) -> faiss.Index:
    """Загружает индекс FAISS из файла."""
    try:
        index = faiss.read_index(index_path)
        logger.info(f"Индекс FAISS загружен из {index_path}")
        return index
    except FileNotFoundError:
        logger.error(f"Файл не найден: {index_path}")
        raise
    except Exception as e:
        logger.error(f"Ошибка загрузки индекса: {str(e)}")
        raise


def load_faiss_index_and_metadata(index_path: str, metadata_path: str) -> Tuple[faiss.Index, List[str]]:
    """
    Загружает индекс FAISS и связанные с ним метаданные.

    Args:
        index_path: Путь к файлу индекса
        metadata_path: Путь к файлу метаданных

    Returns:
        Tuple[faiss.Index, List[str]]: Загруженный индекс и соответствующие метаданные
    """
    try:
        logger.info(f"Загрузка индекса из {index_path}")
        index = faiss.read_index(index_path)

        logger.info(f"Загрузка метаданных из {metadata_path}")
        with open(metadata_path, 'rb') as f:
            texts = pickle.load(f)

        if len(texts) != index.ntotal:
            raise ValueError("Количество текстов не соответствует количеству векторов в индексе")

        logger.info(f"Успешно загружены индекс с {index.ntotal} векторами и {len(texts)} метаданными")
        return index, texts
    except Exception as e:
        logger.error(f"Ошибка загрузки индекса и метаданных: {str(e)}", exc_info=True)
        raise


def query_index(index: faiss.Index, texts: List[str], query_text: str, model: SentenceTransformer, k: int = 5) -> List[
    str]:
    """Запрашивает индекс FAISS и возвращает соответствующие метаданные (тексты)."""
    try:
        logger.info(f"Кодирование запроса: '{query_text}'")
        query_embedding = model.encode([query_text])

        logger.info("Поиск ближайших соседей в индексе FAISS")
        _, indices = index.search(query_embedding, k)

        results = [texts[i] for i in indices[0] if i < len(texts)]
        logger.info(f"Найдено {len(results)} ближайших соседей")
        return results
    except Exception as e:
        logger.error(f"Ошибка при запросе индекса FAISS: {str(e)}")
        raise


def prepare_prompt(question: str, context: List[str], max_context_length: int = 5) -> str:
    """
    Формирует промпт с вопросом и контекстом из базы.

    Args:
        question: Вопрос пользователя
        context: Список строк с контекстом
        max_context_length: Максимальное количество строк контекста для использования

    Returns:
        str: Сформированный промпт
    """
    # Ограничиваем количество строк контекста
    trimmed_context = context[:max_context_length]
    print(trimmed_context)

    if not trimmed_context:
        return f"Вопрос: {question}\nОтвет: В базе данных нет информации по данному вопросу."

    # Объединяем контекст в один текст
    context_str = "\n".join(trimmed_context)

    # Формируем промпт
    return (
        "Используй следующий контекст для ответа на вопрос.\n"
        "Если в контексте нет информации для ответа, сообщи об этом.\n\n"
        f"Контекст:\n{context_str}\n\n"
        f"Вопрос: {question}\n"
        "Ответ:"
    )


def query_ollama(prompt: str, model: str = "llama3.2", timeout: int = 600) -> str:
    """
    Запрашивает Ollama для генерации ответа на основе промпта.

    Args:
        prompt: Промпт для отправки модели
        model: Название модели Ollama
        timeout: Таймаут запроса

    Returns:
        str: Сгенерированный ответ
    """
    url = "http://localhost:8905/api/generate"
    payload = {
        "model": model,
        "prompt": prompt
    }
    headers = {
        "Content-Type": "application/json"
    }

    logger.info(f"Отправка запроса к Ollama, модель: {model}")

    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            stream=True,
            timeout=timeout
        )

        if response.status_code == 200:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if "response" in data:
                            full_response += data["response"]
                    except json.JSONDecodeError:
                        continue  # Игнорируем ошибки декодирования

            logger.info(f"Получен ответ от Ollama: {full_response[:50]}...")
            return full_response
        else:
            logger.error(f"Ошибка запроса к Ollama: {response.status_code}, {response.text}")
            raise Exception(f"Error querying Ollama: {response.status_code}, {response.text}")

    except requests.exceptions.Timeout:
        logger.error("Таймаут при запросе к Ollama")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Сетевая ошибка при запросе к Ollama: {str(e)}")
        raise


def answer_question(question: str) -> str:
    """
    Обрабатывает вопрос пользователя от начала до конца.

    Args:
        question: Вопрос пользователя

    Returns:
        str: Сгенерированный ответ
    """
    try:
        logger.info(f"Обработка вопроса: {question}")

        # Пути к данным
        data_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        path_faiss = os.path.join(data_folder, 'faiss_index.bin')
        path_metadata = os.path.join(data_folder, 'metadata.pkl')

        # Загрузка индекса и метаданных
        index, texts = load_faiss_index_and_metadata(path_faiss, path_metadata)

        # Поиск в FAISS
        model = get_model()
        query_results = query_index(index=index, texts=texts, query_text=question, model=model)

        if not query_results:
            logger.warning("Не найдено релевантного контекста для вопроса")
            return "В базе данных нет информации по данному вопросу."

        # Подготовка промпта и запрос к Ollama
        prompt = prepare_prompt(question, query_results)
        answer = query_ollama(prompt)

        logger.info("Вопрос обработан успешно")
        return answer

    except Exception as e:
        logger.error(f"Ошибка при обработке вопроса: {str(e)}", exc_info=True)
        return "Произошла ошибка при обработке вашего запроса."


def main():
    """Основная функция для обработки вопроса пользователя"""
    question = 'Город Люксембург впервые упоминается в каком году?'
    answer = answer_question(question)
    print(f"Вопрос: {question}")
    print(f"Ответ: {answer}")


if __name__ == "__main__":
    main()