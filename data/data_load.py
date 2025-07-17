import pandas as pd
import json
import requests
import os

def load_data(url):
    # Проверяем наличие файла
    if not os.path.exists("data.json"):
        try:
            # Загружаем данные из URL
            response = requests.get(url)
            response.raise_for_status()  # Проверяем статус ответа
            with open("data.json", "w") as f:
                f.write(response.text)
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"Ошибка при загрузке данных: {e}")
            return None

    # Чтение JSON
    try:
        with open("data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Ошибка при чтении данных: {e}")
        return None


if __name__ == '__main__':
    # Сохраним данные в файл data.csv,
    url = "https://raw.githubusercontent.com/vladislavneon/RuBQ/refs/heads/master/RuBQ_2.0/RuBQ_2.0_paragraphs.json"
    df = load_data(url)
    if df is not None:
        print(df.head())