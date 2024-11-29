import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from clean import Cleaner

PATH_TO_CSV = './16k_Movies.csv'

def openFile():
    file = PATH_TO_CSV
    data = pd.read_csv(file)
    return data

def get_info(df):
    df.info()

def get_count_of_duplicates(df):
    count_duplicates = len(df) - len(df.drop_duplicates())
    print(f"Очищено записей: {count_duplicates}")

# Функция для анализа среднего рейтинга по жанрам
def analyze_genres(df):
    df['Genres'] = df['Genres'].str.split(',')
    df = df.explode('Genres')
    df = df[df['Genres'] != 'Unknown']
    
    genre_ratings = df.groupby('Genres')['Rating'].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    ax = genre_ratings.plot(kind='bar', color='skyblue', edgecolor='black')

    for i, value in enumerate(genre_ratings):
        ax.text(i, value + 0.1, f'{value:.2f}', ha='center', va='bottom', fontsize=9)

    plt.title('Средний рейтинг фильмов по жанрам')
    plt.xlabel('Жанры')
    plt.ylabel('Средний рейтинг')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("Средние рейтинги по жанрам:")
    print(genre_ratings)

# Функция для очистки и преобразования продолжительности
def clean_duration(df):
    def duration_to_minutes(duration):
        match = re.match(r'(\d+)\s*h\s*(\d+)\s*m', str(duration))
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            return hours * 60 + minutes 
        match = re.match(r'(\d+)\s*m', str(duration))
        if match:
            return int(match.group(1))
        return None
    
    df['Duration'] = df['Duration'].apply(duration_to_minutes)
    df = df.dropna(subset=['Duration'])
    return df

# Функция для анализа продолжительности фильмов
def analyze_duration(df):
    df = df[df['Duration'] > 0] 
    lower_percentile = df['Duration'].quantile(0.02)
    upper_percentile = df['Duration'].quantile(0.98)
    df = df[(df['Duration'] >= lower_percentile) & (df['Duration'] <= upper_percentile)]

    plt.figure(figsize=(12, 6))
    sns.histplot(df['Duration'], kde=True, color='skyblue', bins=80)

    plt.title('Распределение продолжительности фильмов')
    plt.xlabel('Продолжительность (минуты)')
    plt.ylabel('Количество фильмов')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print(f"Основные статистики по продолжительности:\n{df['Duration'].describe()}")
    print(f"Средняя продолжительность: {df['Duration'].mean():.2f} минут")

# Функция для очистки и преобразования даты релиза
def extract_year(df):
    df['Year'] = df['Release Date'].str[-4:]
    return df

# Функция для анализа среднего рейтинга фильмов по годам
def analyze_average_rating_by_year(df):
    df_avg_rating_by_year = df.groupby('Year')['Rating'].mean()
    df_avg_rating_by_year = df_avg_rating_by_year.sort_index()

    plt.figure(figsize=(12, 6))
    df_avg_rating_by_year.plot(kind='bar', color='skyblue', edgecolor='black')

    for i, value in enumerate(df_avg_rating_by_year):
        plt.text(i, value + 0.1, f'{value:.2f}', ha='center', va='bottom', fontsize=9)

    plt.title('Средний рейтинг фильмов по годам')
    plt.xlabel('Год')
    plt.ylabel('Средний рейтинг')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print(f"Средний рейтинг по годам:\n{df_avg_rating_by_year}")
    
# Функция для анализа количества фильмов по годам
def analyze_movie_count_by_year(df):
    df = df[df['Year'].astype(int) <= 2022]
    df_movie_count_by_year = df.groupby('Year').size()
    df_movie_count_by_year_smoothed = df_movie_count_by_year.rolling(window=3, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(df_movie_count_by_year.index, df_movie_count_by_year_smoothed.values, marker='o', color='skyblue', linestyle='-', linewidth=2, markersize=5)

    plt.title('Количество фильмов по годам (сглаженное)')
    plt.xlabel('Год')
    plt.ylabel('Количество фильмов (сглаженное)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Количество фильмов по годам:\n{df_movie_count_by_year}")

# Функция для преобразования текста в числовые значения
def transform_text_columns(df):
    # Преобразуем количество жанров
    df['Genre Count'] = df['Genres'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)

    # Преобразуем длительность описания в количество символов
    df['Description Length'] = df['Description'].apply(lambda x: len(str(x)))

    # Преобразуем количество голосов в числовое значение
    df['No of Persons Voted'] = pd.to_numeric(df['No of Persons Voted'], errors='coerce')

    # Преобразуем директора и сценариста в количество уникальных имен
    df['Director Count'] = df['Directed by'].apply(lambda x: len(set(x.split(','))) if isinstance(x, str) else 0)
    df['Writer Count'] = df['Written by'].apply(lambda x: len(set(x.split(','))) if isinstance(x, str) else 0)

    return df

# Функция для анализа корреляции
def analyze_correlation(df):
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df = transform_text_columns(df)

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[numeric_columns].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar=True)
    plt.title('Корреляционная матрица')
    plt.tight_layout()
    plt.show()

    print("Корреляционная матрица:")
    print(corr_matrix)

def main():
    df = openFile()
    cleaner = Cleaner()
    df = cleaner.clean_duration(df)
    df = extract_year(df)
    df = df.dropna(subset=['Year'])
    
    analyze_correlation(df)

if __name__ == '__main__':
    main()