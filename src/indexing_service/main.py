import logging
from src.indexing_service.load_and_save import load_data
from vectorize import vectorize_text, create_faiss_index, save_faiss_index_and_metadata
from src.indexing_service.processing import process_data
import os
import sys
from typing import Any

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../../data/data.log"),  # Логирование в файл
        logging.StreamHandler()  # Логирование в консоль
    ]
)


def validate_data(df: Any) -> bool:
    """Проверяет корректность загруженных данных."""
    if df is None or df.empty:
        logging.error("Данные не загружены или пусты")
        return False

    if 'text' not in df.columns:
        logging.error("Отсутствует необходимая колонка 'text'")
        return False

    if not df['text'].apply(lambda x: isinstance(x, str)).all():
        logging.warning("Некоторые текстовые данные не являются строками")

    return True

def main():
    """Основной процесс создания и сохранения векторного индекса."""
    try:
        # Загрузка данных
        url = "https://raw.githubusercontent.com/vladislavneon/RuBQ/refs/heads/master/RuBQ_2.0/RuBQ_2.0_paragraphs.json "
        logging.info(f"Загрузка данных из {url}")
        df = load_data(url)

        # Проверка данных
        if not validate_data(df):
            logging.error("Данные не прошли валидацию")
            return

        # Обработка данных
        logging.info("Начинаем обработку данных")
        df = process_data(df, 'text')
        texts = df['text'].tolist()
        logging.info(f"Обработано {len(texts)} текстовых фрагментов")

        # Создание эмбеддингов
        logging.info("Создаем эмбеддинги текстовых данных")
        embeddings = vectorize_text(texts)

        # Определяем путь к папке data
        data_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        path_index = os.path.join(data_folder, 'faiss_index.bin')
        metadata_path = os.path.join(data_folder, 'metadata.pkl')

        # Создаем индексы faiss и сохраняем метаданные
        index = create_faiss_index(embeddings)
        save_faiss_index_and_metadata(index=index, texts=texts, index_path=path_index, metadata_path=metadata_path)

        logging.info("Индекс FAISS успешно создан и сохранен.")

    except Exception as e:
        logging.error(f"Критическая ошибка в процессе: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
