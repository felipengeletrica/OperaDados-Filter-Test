import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def filter_heights(dataset, window_size=5, num_values=5, threshold=3):
    df = dataset

    # Filtro de Média Móvel
    df['altura_smoothed'] = df['altura'].rolling(window=window_size, center=True).mean()

    # Descarte de Outliers
    mean = df['altura_smoothed'].mean()
    std_dev = df['altura_smoothed'].std()
    df['altura_filtered'] = df['altura_smoothed'].apply(
        lambda x: x if np.abs(x - mean) <= threshold * std_dev else np.nan
    )

    df['altura_filtered'].fillna(method='ffill', inplace=True)  # Forward fill to handle NaNs
    df['altura_filtered'].fillna(method='bfill', inplace=True)  # Backward fill to handle NaNs

    df_sorted = df.sort_values(by='altura_filtered')
    lowest_heights = df_sorted.head(num_values)
    average_height = lowest_heights['altura_filtered'].mean()
    print(f"A média dos {num_values} menores valores de altura filtrada é: {average_height:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['altura'], label='Valores Originais', marker='o', alpha=0.5)
    plt.plot(df.index, df['altura_filtered'], label='Valores Filtrados', marker='o')
    plt.scatter(lowest_heights.index, lowest_heights['altura_filtered'], color='red', label='Menores Valores', zorder=5)
    plt.title('Valores de Altura Originais e Filtrados')
    plt.xlabel('Índice')
    plt.ylabel('Altura')
    plt.legend()
    plt.show()

def filter_and_plot_heights(dataset, num_values=5):
    df = dataset

    df_sorted = df.sort_values(by='altura')
    lowest_heights = df_sorted.head(num_values)
    average_height = lowest_heights['altura'].mean()
    print(f"A média dos {num_values} menores valores de altura é: {average_height:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['altura'], label='Valores Originais', marker='o')
    plt.scatter(lowest_heights.index, lowest_heights['altura'], color='red', label='Menores Valores', zorder=5)
    plt.title('Valores de Altura Originais e Filtrados')
    plt.xlabel('Índice')
    plt.ylabel('Altura')
    plt.legend()
    plt.show()

data = {
    "altura" : [148.1, 83.9, 147.1, 147.4, 81.4,
        88.9, 81.7, 88.8, 81.2, 81.2,
        88.7, 81.0, 81.4, 146.7, 146.7,
        88.4, 145.4, 145.7, 145.2, 145.2,
        148.1, 83.9, 147.1, 147.4, 81.4,
        88.9, 81.7, 88.8, 81.2, 81.2,
        88.7, 81.0, 81.4, 146.7, 146.7,
        88.4, 145.4, 145.7, 145.2, 145.2,
        148.1, 83.9, 147.1, 147.4, 81.4,
        88.9, 81.7, 88.8, 81.2, 81.2,
        88.7, 81.0, 81.4, 146.7, 146.7,
        88.4, 145.4, 145.7, 145.2, 145.2,
        148.1, 83.9, 147.1, 147.4, 81.4,
        88.9, 81.7, 88.8, 81.2, 81.2,
        88.7, 81.0, 81.4, 146.7, 146.7,
        88.4, 145.4, 145.7, 145.2, 145.2]
}

dataframe = pd.DataFrame(data)

filter_heights(dataframe, window_size=5, num_values=5, threshold=3)
filter_and_plot_heights(dataframe, num_values=5)