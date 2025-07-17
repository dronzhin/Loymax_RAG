import pandas as pd
import re
from data_load import load_data
from data_analysis import analyze_data


# Очистка текста
def clean_text(text):
    """
    Функция очищает текст от лишних символов и пробелов.
    Возвращает очищенный текст.
    """
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s.,—'\"«»]", "", text)  # Сохраняем апострофы и кавычки
    return text

# Загрузка данных
def load_and_process_data():
    """
    Функция загружает данные, обрабатывает пропуски и очищает текст.
    Возвращает обработанный датафрейм.
    """
    df = load_data()

    # Обработка пропусков
    df["text"] = df["text"].fillna("")

    # Группировка по ru_wiki_pageid
    grouped_df = df.groupby("ru_wiki_pageid").agg(
        text=("text", lambda x: ". ".join(x.astype(str)))
    ).reset_index()

    grouped_df["text"] = grouped_df["text"].apply(clean_text)

    return grouped_df

# Сохранение результата
def save_processed_data(grouped_df):
    """
    Функция сохраняет обработанный датафрейм в CSV файл.
    """
    grouped_df.to_csv("combined_data.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    # Логирование
    import logging

    logging.basicConfig(level=logging.INFO)

    try:
        # Загрузка и обработка данных
        processed_df = load_and_process_data()

        # Анализ данных
        analyze_data(processed_df)

        # Сохранение результата
        save_processed_data(processed_df)

    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")