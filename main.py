# Библиотека для работы с матрицами
import numpy as np
# Библиотека для нормализация данных MinMax
from sklearn.preprocessing import MinMaxScaler
# Библиотека для работы с csv файлом
import pandas as pd


# Задание 3. Функция для нормализации данных MinMax и после с сохранением в файл "normalized_data.csv"
def min_max_data_normalize_func():
    data = pd.read_csv('data/kc_house_data.csv', sep=',')
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)

    # Сохранение нормализованных данных в файл "normalized_data.csv"
    df = pd.DataFrame(normalized_data)
    df.to_csv('data/normalized_data.csv', encoding='utf-8', index=False)


# Задание 4. Функция для вычисления минимального, максимального и среднего значения
# для каждого нормализованного столбца (price, bedrooms, sqft_living, waterfront)
def calculate_mathematical_operations_func():
    data = pd.read_csv('data/normalized_data.csv', sep=',', skiprows=[1])

    min_values = data.min()
    max_values = data.max()
    mean_values = data.mean()

    print(f"Минимальное значение:\n{min_values.to_string()}")
    print(f"Максимальное значение:\n{max_values.to_string()}")
    print(f"Среднее значение:\n{mean_values.to_string()}\n")


# 5.1 Функция для расчёта ошибки RMSE
def rmse_calculation_func(normalized_data, prediction_data):
    global actual, predicted

    for i in range(normalized_data.shape[1]):
        initial_column = normalized_data[:, i]
        initial_data_column = (initial_column - np.mean(initial_column)) / np.std(initial_column)
        normalized_data[:, i] = initial_data_column

        normalized_column = prediction_data[:, i]
        normalized_data_column = (normalized_column - np.mean(normalized_column)) / np.std(normalized_column)
        prediction_data[:, i] = normalized_data_column

        minimum = np.min(initial_data_column)
        maximum = np.max(initial_data_column)

        actual = (initial_data_column - minimum) / (maximum - minimum)
        predicted = (normalized_data_column - minimum) / (maximum - minimum)

    return np.sqrt(np.mean((actual - predicted) ** 2))


# 5.2 Функция расчёта градиента функции ошибки RMSE
def run_task_five_two(initial_data, normalized_data):
    global actual, predicted
    gradient_array = np.zeros((initial_data.shape[1],))

    for i in range(initial_data.shape[1]):
        initial_column = initial_data[:, i]
        initial_data_column = (initial_column - np.mean(initial_column)) / np.std(initial_column)
        initial_data[:, i] = initial_data_column

        normalized_column = normalized_data[:, i]
        normalized_data_column = (normalized_column - np.mean(normalized_column)) / np.std(normalized_column)
        normalized_data[:, i] = normalized_data_column

        minimum = np.min(initial_data_column)
        maximum = np.max(initial_data_column)

        actual = (initial_data_column - minimum) / (maximum - minimum)
        predicted = (normalized_data_column - minimum) / (maximum - minimum)
        gradient_array[i] = -2 * np.mean((actual - predicted) * ((maximum - minimum) / np.std(initial_column)) * (
                normalized_column - np.mean(normalized_column)))

    return gradient_array


def run_task_five():
    initial_data = np.array(pd.read_csv('data/kc_house_data.csv', sep=',', skiprows=[1]))
    normalized_data = np.array(pd.read_csv('data/normalized_data.csv', sep=',', skiprows=[1]))

    rmse = rmse_calculation_func(initial_data, normalized_data)
    print(f"Результат функции расчёта ошибки RMSE: {rmse}")

    # gradient_array = run_task_five_two(initial_data, normalized_data)
    # print(f"Результат функции расчёта градиента ошибки: {gradient_array}")


# 6 Написать функцию градиентного спуска для коэффициентов линейной регрессии.
# Во время градиентного спуска шаг рассчитывать по написанной ранее производной функции ошибки.
def run_task_six():
    initial_data = np.loadtxt("data/kc_house_data.csv", delimiter=',', skiprows=1)
    initial_data = np.array(initial_data, dtype=np.float32)

    normalized_data = np.loadtxt("data/normalized_data.csv", delimiter=',')
    normalized_data = np.array(normalized_data, dtype=np.float32)

    gradient_array = run_task_five_two(initial_data, normalized_data)



if __name__ == '__main__':
    min_max_data_normalize_func()
    calculate_mathematical_operations_func()
    run_task_five()
    # # run_task_six()
