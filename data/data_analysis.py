import pandas as pd
from data_load import load_data


def analyze_data(df):
    """
    Функция для анализа данных и вывода ключевой информации.
    :param df: Датафрейм с данными
    """

    # Количество уникальных значений по столбцам
    unique_values = df.nunique()
    print("Количество уникальных значений в каждом столбце:")
    print(unique_values)

    # Минимальная и максимальная длина строк для строковых столбцов
    for col in df.select_dtypes(include='object').columns:
        min_len = df[col].str.len().min()
        max_len = df[col].str.len().max()
        print(f"Столбец '{col}': Минимальная длина: {min_len}, Максимальная длина: {max_len}")


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