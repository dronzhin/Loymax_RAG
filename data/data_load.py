import pandas as pd
import json
import requests
import os

# Константа для загрузки данных
URL = "https://raw.githubusercontent.com/vladislavneon/RuBQ/refs/heads/master/RuBQ_2.0/RuBQ_2.0_paragraphs.json"
FILENAME = "data.json"


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
            response = requests.get(url)
            response.raise_for_status()  # Проверяем статус ответа
            with open(FILENAME, "w", encoding="utf-8") as f:
                f.write(response.text)
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"Ошибка при загрузке данных: {e}")
            return None

    # Чтение JSON
    try:
        with open(FILENAME, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Ошибка при чтении данных: {e}")
        return None


if __name__ == '__main__':
    # Сохраним данные в файл data.csv,
    df = load_data()
    if df is not None:
        print(df.head())