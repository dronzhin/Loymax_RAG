import pandas as pd
import re
from data_load import load_data
from data_analysis import analyze_data
from joblib import Parallel, delayed
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Очистка текста
def clean_text(text):
    """Очистка текста от лишних символов и пробелов."""
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"[^\w\s.,—'\"«»]", "", text)


# Загрузка и предобработка данных
def load_and_process_data():
    """Загрузка, группировка и очистка данных."""
    logger.info("Начало загрузки данных")
    df = load_data()
    logger.info(f"Загружено {len(df)} записей")

    # Проверка кодировки и битых символов
    df = check_and_fix_utf8_validity(df)
    df = check_and_fix_replacement_chars(df)
    df = check_non_printable_chars(df)

    logger.info("Группировка по ru_wiki_pageid")
    grouped_df = df.groupby("ru_wiki_pageid").agg(
        text=("text", lambda x: ". ".join(x.astype(str)))
    ).reset_index(drop=True)

    logger.info("Очистка текста")
    grouped_df["text"] = Parallel(n_jobs=-1)(
        delayed(clean_text)(text) for text in grouped_df["text"]
    )

    logger.info("Создание новых uid")
    grouped_df["uid"] = range(len(grouped_df))
    #grouped_df.drop("ru_wiki_pageid", axis=1, inplace=True)

    logger.info("Обработка данных завершена")
    return grouped_df


# Проверка дубликатов
def check_for_duplicates(data):
    """Проверка и удаление дубликатов."""
    logger.info("Проверка дубликатов")
    duplicates = data[data["text"].duplicated()]

    if not duplicates.empty:
        logger.warning(f"Найдено {len(duplicates)} дубликатов")
        logger.debug("Пример дубликатов:\n%s", duplicates.head())
        cleaned = data.drop_duplicates("text", keep="first").reset_index(drop=True)
        logger.info(f"Удалено {len(data) - len(cleaned)} дубликатов")
        return cleaned
    logger.info("Дубликаты не найдены")
    return data


# Разбиение длинного текста
def split_text_by_sentences(text, max_length=20000):
    """Разбиение текста на части по предложениям."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += f"{sentence} "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = f"{sentence} "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# Параллельное разбиение текста
def process_dataframe(df, max_length=20000):
    """Разбиение длинных текстов на части."""
    logger.info(f"Разбиение текстов длиннее {max_length} символов")

    result = []
    for _, row in df.iterrows():
        if len(row["text"]) > max_length:
            chunks = split_text_by_sentences(row["text"], max_length)
            for i, chunk in enumerate(chunks):
                result.append({"uid": row["uid"], "text": chunk})
        else:
            result.append({"uid": row["uid"], "text": row["text"]})

    new_df = pd.DataFrame(result)
    new_df["uid"] = range(len(new_df))  # Обновление uid
    logger.info(f"Текст разбит на {len(new_df) - len(df)} дополнительных частей")
    return new_df


# Проверка кодировки и битых символов
def check_and_fix_utf8_validity(df, column="text"):
    """
    Проверка и исправление валидности UTF-8.
    """
    logger.info("Проверка и исправление валидности UTF-8")

    def is_valid_utf8(text):
        try:
            text.encode("utf-8").decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False

    def fix_utf8(text):
        try:
            return text.encode("utf-8", errors='replace').decode("utf-8")
        except Exception as e:
            logger.error(f"Ошибка при исправлении UTF-8: {e}")
            return ""

    # Проверка валидности UTF-8
    df["invalid_utf8"] = df[column].apply(lambda x: not is_valid_utf8(x))

    # Исправление невалидных строк
    invalid = df[df["invalid_utf8"]]
    if not invalid.empty:
        logger.warning(f"Найдено {len(invalid)} записей с невалидным UTF-8")
        logger.debug("Исправление невалидных строк")

        df.loc[df["invalid_utf8"], column] = df.loc[df["invalid_utf8"], column].apply(fix_utf8)

        # Удаление временного столбца
        df.drop("invalid_utf8", axis=1, inplace=True)

    return df

def check_and_fix_replacement_chars(df, column="text"):
    """
    Проверка и исправление символов замены.
    """
    logger.info("Проверка и исправление символов замены")

    # Исправление символов замены
    def fix_replacement_chars(text):
        # Заменяем символы замены на пустую строку
        return text.replace('\ufffd', '')

    # Проверка наличия символов замены
    pattern = re.compile(r"\ufffd")
    df["has_replacement"] = df[column].apply(lambda x: bool(pattern.search(x)))

    # Исправление строк с символами замены
    issues = df[df["has_replacement"]]
    if not issues.empty:
        logger.warning(f"Найдено {len(issues)} записей с символами замены")
        logger.debug("Исправление строк с символами замены")

        df.loc[df["has_replacement"], column] = df.loc[df["has_replacement"], column].apply(fix_replacement_chars)

        # Удаление временного столбца
        df.drop("has_replacement", axis=1, inplace=True)

    return df



def check_non_printable_chars(df, column="text"):
    """Проверка наличия непечатаемых символов."""
    logger.info("Проверка непечатаемых символов")
    pattern = re.compile(r"[^\x20-\x7E\x80-\xFF\u0400-\u04FF]")
    df["has_non_printable"] = df[column].apply(lambda x: bool(pattern.search(x)))
    issues = df[df["has_non_printable"]]

    if not issues.empty:
        logger.warning(f"Найдено {len(issues)} записей с непечатаемыми символами")
        logger.debug("Примеры с непечатаемыми символами:\n%s", issues.head())
    return df


# Сохранение результата
def save_processed_data(df):
    """Сохранение обработанного DataFrame в CSV."""
    logger.info("Сохранение данных в файл combined_data.csv")
    df.to_csv("combined_data.csv", index=False, encoding="utf-8-sig")
    logger.info("Данные успешно сохранены")


# Основной процесс
if __name__ == "__main__":
    try:

        # Загрузка и обработка
        df = load_and_process_data()

        # Проверка дубликатов
        df = check_for_duplicates(df)

        # Разбиение длинных текстов
        df = process_dataframe(df)

        # Проверка кодировки и битых символов
        df = check_and_fix_utf8_validity(df)
        df = check_and_fix_replacement_chars(df)
        df = check_non_printable_chars(df)

        # Анализ и сохранение
        analyze_data(df)
        save_processed_data(df)

    except Exception as e:
        logger.error("Критическая ошибка: %s", e, exc_info=True)