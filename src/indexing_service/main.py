import logging
import faiss
from src.indexing_service.load_and_save import load_data
from vectorize import vectorize_text, create_faiss_index
from src.indexing_service.processing import process_data
import os

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../../data/data.log"),  # Логирование в файл
        logging.StreamHandler()  # Логирование в консоль
    ]
)

def main():
    url = "https://raw.githubusercontent.com/vladislavneon/RuBQ/refs/heads/master/RuBQ_2.0/RuBQ_2.0_paragraphs.json"
    df = load_data(url)
    df = process_data(df, 'text')

    texts = df['text'].tolist()
    embeddings = vectorize_text(texts)
    index = create_faiss_index(embeddings)

    # Определяем путь к папке data
    data_folder = os.path.join(os.path.dirname(__file__), '..', '..','data')
    path_index = os.path.join(data_folder, 'faiss_index.bin')

    faiss.write_index(index, path_index)
    logging.info("Индекс FAISS успешно создан и сохранен.")

if __name__ == "__main__":
    main()
