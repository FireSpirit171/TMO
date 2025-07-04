# Лабораторная работа №9: Кластеризация и снижение размерности на примере датасета вина

## 📌 Цель работы

Изучить методы снижения размерности (PCA, t-SNE) и алгоритмы кластеризации (KMeans, агломеративная кластеризация, Gaussian Mixture Model) на реальных данных. Сравнить качество кластеризации на исходных признаках и на пониженных измерениях.

## 🧠 Задача

- Применить масштабирование признаков и методы снижения размерности PCA и t-SNE к датасету.
- Кластеризовать данные с помощью KMeans, агломеративной кластеризации и GMM.
- Оценить качество кластеров с помощью метрик: Silhouette Score, Calinski-Harabasz, Davies-Bouldin.
- Визуализировать результаты кластеризации на трех представлениях данных (исходные признаки, PCA, t-SNE).

## 🧹 Датасет

Данные о качестве красного вина: [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)

Использован файл `winequality-red.csv`, содержащий 1599 записей и 11 химических признаков вина (pH, кислотность, сахар, алкоголь и др.). Колонка `quality` была исключена из анализа.

## 🧹 Предобработка данных

1. Загрузка и масштабирование признаков StandardScaler.
2. Снижение размерности:
   - PCA с 2 компонентами — для линейного уменьшения размерности.
   - t-SNE с 2 компонентами — для нелинейного отображения данных.
3. Определение главных признаков, влияющих на компоненты PCA.

## 🧪 Модели и кластеризация

Использованы три метода:

- **KMeans** — алгоритм центроидного разбиения, оптимизирующий внутрикластерное расстояние.
- **Agglomerative Clustering** — иерархический метод, последовательно объединяющий ближайшие кластеры.
- **Gaussian Mixture Model (GMM)** — вероятностная модель, описывающая данные как смесь гауссианов, позволяет кластерам перекрываться.

Все методы настроены на 3 кластера (кроме GMM — 3 компоненты).

## 📊 Метрики оценки качества

- **Silhouette Score** — насколько хорошо объекты соотносятся с собственным кластером по сравнению с соседними.
- **Calinski-Harabasz** — отношение межкластерной дисперсии к внутрикластерной.
- **Davies-Bouldin** — мера сходства кластеров (меньше — лучше).

## 📈 Визуализация

- Отдельные графики для каждого метода и каждого представления данных.
- Для PCA оси подписаны с указанием признаков, наиболее сильно влияющих на компоненты.
- Для t-SNE и исходных данных оси обозначены как Dim 1 и Dim 2.

## ✅ Выводы

- Снижение размерности облегчает визуализацию, при этом кластеризация на PCA и t-SNE может выявлять структуру данных.
- GMM показывает более гибкое моделирование кластеров, чем жёсткие методы.
- Метрики и визуализации помогают сравнить качество различных алгоритмов на разных представлениях.

## 📎 Используемые библиотеки

- `pandas`, `numpy` — работа с данными;
- `matplotlib`, `seaborn` — визуализация;
- `sklearn` — масштабирование, PCA, t-SNE, кластеризация, метрики;
- `sklearn.mixture` — GaussianMixture.
