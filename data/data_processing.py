import pandas as pd
import re
from data_load import load_data, save_data_to_csv
from data_analysis import analyze_data
from joblib import Parallel, delayed
import logging

# Создаем логгер для текущего модуля
logger = logging.getLogger(__name__)

# Очистка текста
def clean_text(text):
    """Очистка текста от лишних символов и пробелов."""
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"[^\w\s.,—'\"«»]", "", text)


def filter_dataframe_by_text_length(df: pd.DataFrame, column: str = 'text', min_text_length: int = 3) -> pd.DataFrame:
    """
    Функция для фильтрации строк DataFrame на основе длины текста в указанной колонке с логированием.

    :param df: DataFrame, который нужно отфильтровать
    :param column: Название колонки для проверки длины текста, по умолчанию 'text'
    :param min_text_length: Минимальная длина текста, строки с меньшей длиной будут удалены, по умолчанию 4
    :return: Отфильтрованный DataFrame
    """
    logger.info(
        "Начало фильтрации DataFrame по длине текста в колонке: %s, минимальная длина: %d",
        column, min_text_length
    )

    # Создаем копию DataFrame, чтобы избежать изменения исходного DataFrame
    filtered_df = df.copy()

    # Проверяем наличие колонки в DataFrame
    if column not in filtered_df.columns:
        logger.error("Колонка '%s' не найдена в DataFrame", column)
        return filtered_df

    # Фильтруем строки на основе длины текста в указанной колонке
    initial_count = len(filtered_df)
    filtered_df = filtered_df[filtered_df[column].str.len().fillna(0) >= min_text_length]
    final_count = len(filtered_df)

    logger.info(
        "Фильтрация по колонке '%s': удалено %d строк, осталось %d строк",
        column, initial_count - final_count, final_count
    )

    return filtered_df


# Загрузка и предобработка данных
def process_data(df, column: str = 'text'):

    # Проверка кодировки и битых символов
    df = check_and_fix_utf8_validity(df, column=column)
    df = check_and_fix_replacement_chars(df, column=column)
    df = check_and_del_non_printable_chars(df, column=column)

    # Фильтрации строк DataFrame на основе длины текста
    df = filter_dataframe_by_text_length(df, column = column)

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

    # Разбиение длинных текстов
    grouped_df = process_dataframe(grouped_df)

    # Проверка дубликатов
    grouped_df = check_for_duplicates(grouped_df)

    logger.info("Создание новых uid")
    grouped_df["uid"] = range(len(grouped_df))

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

    # Нахождение оптимального размера чанка (среднее значение близкое, но не более max_length)
    max_optim_length = len(text)//(len(text)//max_length + 1)
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_optim_length:
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


def check_and_fix_utf8_validity(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Проверка и исправление валидности UTF-8 в указанной колонке DataFrame.

    :param df: DataFrame, который нужно проверить и исправить
    :param column: Название колонки для проверки и исправления
    :return: DataFrame с исправленными строками
    """
    logger.info("Проверка и исправление валидности UTF-8 в колонке '%s'", column)

    def is_valid_utf8(text: str) -> bool:
        """
        Проверяет, является ли строка валидной в кодировке UTF-8.

        :param text: Строка для проверки
        :return: True, если строка валидная в UTF-8, иначе False
        """
        try:
            text.encode('utf-8').decode('utf-8')
            return True
        except UnicodeError:
            return False

    def fix_utf8(text: str) -> str:
        """
        Исправляет строку, заменяя невалидные символы.

        :param text: Строка для исправления
        :return: Исправленная строка
        """
        try:
            return text.encode('utf-8', errors='replace').decode('utf-8')
        except Exception as e:
            logger.error(f"Ошибка при исправлении UTF-8: {e}")
            return ""

    # Проверка валидности UTF-8
    df["invalid_utf8"] = df[column].apply(lambda x: not is_valid_utf8(str(x)))

    # Исправление невалидных строк
    invalid = df[df["invalid_utf8"]]
    if not invalid.empty:
        logger.warning("Найдено %d записей с невалидным UTF-8", len(invalid))
        logger.debug("Исправление невалидных строк")
        df.loc[df["invalid_utf8"], column] = df.loc[df["invalid_utf8"], column].apply(fix_utf8)

    # Удаление временного столбца
    df.drop("invalid_utf8", axis=1, inplace=True)

    logger.info("Исправление завершено")
    return df


def check_and_fix_replacement_chars(df, column: str):
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



def check_and_del_non_printable_chars(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Проверка и удаление непечатаемых символов в указанной колонке DataFrame.

    :param df: DataFrame, который нужно проверить и очистить
    :param column: Название колонки для проверки и очистки, по умолчанию 'text'
    :return: DataFrame с удаленными непечатаемыми символами
    """
    logger.info("Проверка и удаление непечатаемых символов в колонке '%s'", column)

    # Паттерн для поиска непечатаемых символов
    pattern = re.compile(r'[^\x20-\x7E\x80-\xFF\u0400-\u04FF]')

    # Проверка наличия непечатаемых символов
    df["has_non_printable"] = df[column].apply(lambda x: bool(pattern.search(str(x))))
    issues = df[df["has_non_printable"]]

    if not issues.empty:
        logger.warning("Найдено %d записей с непечатаемыми символами", len(issues))
        logger.debug("Примеры с непечатаемыми символами:\n%s", issues.head())

        # Удаление непечатаемых символов
        df[column] = df[column].apply(lambda x: pattern.sub('', str(x)))
        logger.info("Непечатаемые символы удалены из колонки '%s'", column)
    else:
        logger.info("Непечатаемые символы не найдены в колонке '%s'", column)

    # Удаление временной колонки
    df.drop(columns=["has_non_printable"], inplace=True)

    return df


# Основной процесс
if __name__ == "__main__":

    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    column = 'text'

    try:
        # Загрузка
        df = load_data()
        # Обработка
        df = process_data(df)

        # Анализ и сохранение
        analyze_data(df)
        save_data_to_csv(df)

    except Exception as e:
        logger.error("Критическая ошибка: %s", e, exc_info=True)