import pandas as pd
import json
import requests
import os
import logging

# Создаем логгер для текущего модуля
logger = logging.getLogger(__name__)

# Константа для загрузки данных
URL = "https://raw.githubusercontent.com/vladislavneon/RuBQ/refs/heads/master/RuBQ_2.0/RuBQ_2.0_paragraphs.json"
FILENAME = "../../data/data.json"

def load_data(url=URL):
    """
    Загружает данные из URL и сохраняет в файл data.json.
    :param url: Ссылка на источник данных (по умолчанию - URL)
    :return: Датафрейм с данными, если загрузка успешна, иначе None
    """
    # Проверяем наличие файла
    if not os.path.exists(FILENAME):
        try:
            # Загружаем данные из URL
            logger.info("Загрузка данных из URL...")
            response = requests.get(url)
            response.raise_for_status()  # Проверяем статус ответа
            with open(FILENAME, "w", encoding="utf-8") as f:
                f.write(response.text)
            logger.info("Данные успешно загружены и сохранены в файл.")
        except requests.RequestException as e:
            logger.error(f"Ошибка при загрузке данных: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка при декодировании JSON: {e}")
            return None

    # Чтение JSON
    try:
        logger.info("Чтение данных из файла...")
        with open(FILENAME, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        logger.info("Данные успешно прочитаны и преобразованы в DataFrame.")
        logger.info(f"Загружено {len(df)} записей")
        return df
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка при декодировании JSON: {e}")
        return None
    except FileNotFoundError as e:
        logger.error(f"Файл не найден: {e}")
        return None

# Сохранение результата
def save_data_to_csv(df):
    """Сохранение обработанного DataFrame в CSV."""
    logger.info("Сохранение данных в файл combined_data.csv")
    df.to_csv("combined_data.csv", index=False, encoding="utf-8-sig")
    logger.info("Данные успешно сохранены")

if __name__ == '__main__':
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Сохраним данные в файл data.csv
    df = load_data()
    if df is not None:
        print(df.head())
