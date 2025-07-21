import logging
from data_load import load_data

# Константа для загрузки данных
URL = "https://raw.githubusercontent.com/vladislavneon/RuBQ/refs/heads/master/RuBQ_2.0/RuBQ_2.0_paragraphs.json"

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data.log"),  # Логирование в файл
        logging.StreamHandler()  # Логирование в консоль
    ]
)

df = load_data()