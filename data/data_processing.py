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
    Функция загружает данные, очищает текст, группирует по uid,
    создает новый uid и сохраняет результат.
    Возвращает обработанный датафрейм.
    """
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


# Функция для разбиения текста примерно равные  на части но не более максимальной длины
def split_text_by_sentences(text, max_length=30000):
    # Разбиваем текст на предложения по знакам .!?
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    # Определяем оптимальное количество максимальной длины
    count_part = len(text)/max_length + 1
    optim_max_length = len(text)//count_part

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= optim_max_length:
            current_chunk += (sentence + " ")
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# Функция разбивающая DataFrame по максимальной длине
def process_dataframe(df, max_length=30000):
    result = []

    for _, row in df.iterrows():
        uid = row['uid']
        text = row['text']

        if len(text) > max_length:
            chunks = split_text_by_sentences(text, max_length)
            for i, chunk in enumerate(chunks):
                result.append({'uid': uid, 'text': chunk})
        else:
            result.append({'uid': uid, 'text': text})

    df = pd.DataFrame(result)

    # Создание нового uid
    df['uid'] = range(len(df))  # Новые uid от 0 до len(grouped_df)

    return df

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

        # Проверка на дубликаты и их удаление
        processed_df = check_for_duplicates(processed_df)

        processed_df = process_dataframe(processed_df, max_length=20000)

        # Анализ данных
        analyze_data(processed_df)

        # Сохранение результата
        save_processed_data(processed_df)
        check_for_duplicates(processed_df)

    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")