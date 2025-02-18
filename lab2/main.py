import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Загрузка данных
df = pd.read_csv('lab2/best-selling-books.csv')

# Обработка пропусков: заменим NaN в Genre на 'Unknown'
df['Genre'] = df['Genre'].fillna('Unknown')

# Разделение жанров на несколько
df['Genre'] = df['Genre'].apply(lambda x: x.split(', ') if isinstance(x, str) else ['Unknown'])

# Уплощение списка жанров
df_exploded = df.explode('Genre')

# Фильтрация жанров, встречающихся 2 и более раз (исключая 'Unknown')
genre_counts = df_exploded['Genre'].value_counts()
df_exploded = df_exploded[df_exploded['Genre'].isin(genre_counts[genre_counts >= 2].index) & (df_exploded['Genre'] != 'Unknown')]

# Визуализация до обработки
plt.figure(figsize=(10, 5))
sns.histplot(df['Approximate sales in millions'], bins=20, kde=True)
plt.title('Распределение продаж до масштабирования')
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(y=df_exploded['Genre'], order=df_exploded['Genre'].value_counts().index)
plt.title('Распределение жанров до кодирования')
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(y=df['Original language'], order=df['Original language'].value_counts().index)
plt.title('Распределение языков до кодирования')
plt.show()

# Кодирование категориальных признаков
label_encoder_genre = LabelEncoder()
label_encoder_language = LabelEncoder()

# Кодируем жанры (исключаем 'Unknown' уже на этом этапе)
df_exploded['Genre'] = label_encoder_genre.fit_transform(df_exploded['Genre'])
df['Original language'] = label_encoder_language.fit_transform(df['Original language'])

# Масштабирование числовых данных
scaler = StandardScaler()
df[['Approximate sales in millions']] = scaler.fit_transform(df[['Approximate sales in millions']])

# Визуализация после обработки
plt.figure(figsize=(10, 5))
sns.histplot(df['Approximate sales in millions'], bins=20, kde=True)
plt.title('Распределение продаж после масштабирования')
plt.show()

# Расшифровка чисел для жанров
print("\nРасшифровка кодированных значений жанров:")
for i, label in enumerate(label_encoder_genre.classes_):
    print(f'{i} - {label}')

plt.figure(figsize=(10, 5))
sns.countplot(y=df_exploded['Genre'], order=df_exploded['Genre'].value_counts().index)
plt.title('Распределение жанров после кодирования')
plt.show()

# Расшифровка чисел для языка
print("\nРасшифровка кодированных значений языков:")
for i, label in enumerate(label_encoder_language.classes_):
    print(f'{i} - {label}')

plt.figure(figsize=(10, 5))
sns.countplot(y=df['Original language'], order=df['Original language'].value_counts().index)
plt.title('Распределение языков после кодирования')
plt.show()

# Проверка изменений
print(df.head())
