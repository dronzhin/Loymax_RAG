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
    df = load_data()

    # Группировка по ru_wiki_pageid и сохранение оригинальных uid
    grouped_df = df.groupby("ru_wiki_pageid").agg({
        "text": lambda x: ". ".join(x.astype(str))
    }).reset_index()

    # Очистка текста
    grouped_df["text"] = grouped_df["text"].apply(clean_text)

    # Создание нового uid
    grouped_df['uid'] = range(len(grouped_df))  # Новые uid от 0 до len(grouped_df)

    # Удаление ru_wiki_pageid, если он не нужен
    grouped_df.drop("ru_wiki_pageid", axis=1, inplace=True)

    # Поменять местами столбцы
    grouped_df = grouped_df[['uid', 'text']]

    return grouped_df



def check_for_duplicates(data):
    """
    Функция проверяет на дубликаты и удаляет их, оставляя первую копию.
    Возвращает обработанный датафрейм без дубликатов.
    """
    duplicates = data[data['text'].duplicated()]

    if not duplicates.empty:
        logging.info("Найдены дубликаты:")
        print(duplicates)

        cleaned_data = data.drop_duplicates(subset=['text'], keep='first')
        logging.info(f"\nУдалено дубликатов: {len(data) - len(cleaned_data)}")
        return cleaned_data
    else:
        logging.info("Дубликатов не найдено.")
        return data

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

        # Проверка на дупликаты и их удаление
        processed_df = check_for_duplicates(processed_df)

        # Анализ данных
        analyze_data(processed_df)

        # Сохранение результата
        save_processed_data(processed_df)
        check_for_duplicates(processed_df)

    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")