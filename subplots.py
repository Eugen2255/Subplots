import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
import numpy as np


def load_diabetes_data():
    """
    Загружает и возвращает данные о диабете.
    
    Returns:
        tuple: Кортеж содержащий:
            - X (numpy.ndarray): Матрица признаков
            - y (numpy.ndarray): Целевая переменная
            - feature_names (numpy.ndarray): Названия признаков
    """
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    feature_names = diabetes.feature_names
    return X, y, feature_names


def create_target_distribution_plots(axes, y):
    """
    Создает графики распределения целевой переменной.
    
    Args:
        axes (numpy.ndarray): Массив осей для построения графиков
        y (numpy.ndarray): Целевая переменная
    """
    # 1. Гистограмма целевой переменной (y)
    axes[0, 0].hist(y, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Гистограмма целевой переменной')
    axes[0, 0].set_xlabel('Прогрессирование заболевания')
    axes[0, 0].set_ylabel('Частота')
    axes[0, 0].grid(alpha=0.3)

    # 2. Boxplot целевой переменной
    axes[0, 1].boxplot(y, vert=True, patch_artist=True)
    axes[0, 1].set_title('Boxplot целевой переменной')
    axes[0, 1].set_ylabel('Прогрессирование заболевания')
    axes[0, 1].grid(alpha=0.3)

    # 3. KDE plot для целевой переменной
    sns.kdeplot(y, ax=axes[1, 0], fill=True, color='orange', alpha=0.7)
    axes[1, 0].set_title('KDE Plot целевой переменной')
    axes[1, 0].set_xlabel('Прогрессирование заболевания')
    axes[1, 0].set_ylabel('Плотность')
    axes[1, 0].grid(alpha=0.3)


def create_feature_distribution_plot(axes, X, feature_index=2, feature_name='BMI'):
    """
    Создает график распределения признака с KDE.
    
    Args:
        axes (matplotlib.axes.Axes): Ось для построения графика
        X (numpy.ndarray): Матрица признаков
        feature_index (int): Индекс признака для визуализации (по умолчанию 2 - BMI)
        feature_name (str): Название признака для подписей (по умолчанию 'BMI')
    """
    axes.hist(X[:, feature_index], bins=30, alpha=0.7, color='lightgreen', 
              edgecolor='black', density=True)
    sns.kdeplot(X[:, feature_index], ax=axes, color='red', linewidth=2)
    axes.set_title(f'Распределение {feature_name} с KDE')
    axes.set_xlabel(f'{feature_name} (индекс массы тела)')
    axes.set_ylabel('Плотность')
    axes.grid(alpha=0.3)


def print_dataset_info(X, y, feature_names):
    """
    Выводит основную информацию о наборе данных.
    
    Args:
        X (numpy.ndarray): Матрица признаков
        y (numpy.ndarray): Целевая переменная
        feature_names (numpy.ndarray): Названия признаков
    """
    print(f"Размерность данных: {X.shape}")
    print(f"Количество признаков: {len(feature_names)}")
    print(f"Признаки: {list(feature_names)}")
    print(f"Диапазон целевой переменной: [{y.min():.1f}, {y.max():.1f}]")


def main():
    """
    Основная функция для выполнения анализа данных о диабете.
    Загружает данные, создает визуализации и выводит информацию о наборе данных.
    """
    # Загружаем данные
    X, y, feature_names = load_diabetes_data()
    
    # Создаем фигуру с сеткой 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Анализ данных Diabetes Dataset', fontsize=16, fontweight='bold')
    
    # Создаем графики распределения целевой переменной
    create_target_distribution_plots(axes, y)
    
    # Создаем график распределения признака BMI
    create_feature_distribution_plot(axes[1, 1], X, feature_index=2, feature_name='BMI')
    
    # Настраиваем отступы и layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    # Выводим информацию о данных
    print_dataset_info(X, y, feature_names)


if __name__ == "__main__":
    main()