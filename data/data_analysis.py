import pandas as pd
from data_load import load_data


def analyze_data(df):
    """
    Функция для анализа данных.

    :param df: Датафрейм с данными
    """
    # 2. Проверка пропусков
    print("Количество пропусков в каждом столбце:")
    print(df.isnull().sum())
    print("\n")

    # 3. Количество уникальных значений по столбцам
    print("Количество уникальных значений в каждом столбце:")
    print(df.nunique())
    print("\n")

    # 4. Минимальная и максимальная длина строк по столбцам (только для строковых столбцов)
    for col in df.select_dtypes(include='object').columns:
        print(f"Столбец '{col}':")
        min_len = df[col].str.len().min()
        max_len = df[col].str.len().max()
        print(f"  Минимальная длина: {min_len}, Максимальная длина: {max_len}")

    # 2. Расчёт длины текста
    df["text_length"] = df["text"].str.len()  # Добавление столбца с длиной текста

    # 3. Сортировка по возрастанию длины текста
    df_sorted = df.sort_values(by="text_length").reset_index(drop=True)

    # 4. Вывод первых 50 записей с минимальным текстом
    print("50 записей с минимальным текстом:")
    print(df_sorted[["text", "text_length"]].head(50))
    df.drop("text_length", axis=1, inplace=True)


def main():
    """
    Основная функция для выполнения анализа данных.
    """
    # Загрузка данных
    df = load_data()

    # Проверка на наличие данных в таблице
    if df.empty:
        print("Таблица пустая. Проверьте, что данные загружены корректно.")
        return

    # Анализ данных
    analyze_data(df)


if __name__ == "__main__":
    main()