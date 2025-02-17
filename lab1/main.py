import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
df = pd.read_csv('lab1/NetflixOriginals.csv', encoding='latin1')

df['Premiere'] = pd.to_datetime(df['Premiere'])
df['Year'] = df['Premiere'].dt.year

# Гистограмма IMDB рейтингов
plt.figure(figsize=(10, 6))
sns.histplot(df['IMDB Score'], bins=20, kde=True)
plt.title('Распределение IMDB рейтингов')
plt.xlabel('IMDB Score')
plt.ylabel('Частота')
plt.show()

# Разделение жанров и создание словаря
genre_dict = {}
for index, row in df.iterrows():
    genres = row['Genre'].split('/')
    for genre in genres:
        if genre in genre_dict:
            genre_dict[genre] += 1
        else:
            genre_dict[genre] = 1

# Преобразуем словарь в DataFrame для удобства визуализации
genre_df = pd.DataFrame(list(genre_dict.items()), columns=['Genre', 'Count'])

# Оставляем только жанры с более чем 5 фильмами
filtered_genre_df = genre_df[genre_df['Count'] > 5]

# Визуализация количества фильмов по жанрам (более 5 фильмов)
plt.figure(figsize=(12, 8))
sns.barplot(x='Count', y='Genre', data=filtered_genre_df.sort_values(by='Count', ascending=False))
plt.title('Количество фильмов по жанрам (более 5 фильмов)')
plt.xlabel('Количество фильмов')
plt.ylabel('Жанр')
plt.show()

# Выбираем только числовые признаки для корреляции
numeric_df = df[['IMDB Score', 'Runtime', 'Year']]

# Кодируем жанр в числовой формат (например, с помощью category codes)
df['Genre'] = df['Genre'].astype('category').cat.codes
numeric_df['Genre'] = df['Genre']

# Построим корреляционную матрицу
correlation_matrix = numeric_df.corr()
print("Корреляционная матрица:")
print(correlation_matrix)

# Тепловая карта корреляции
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Тепловая карта корреляции')
plt.show()
