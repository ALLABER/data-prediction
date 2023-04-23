import csv

import numpy as np
from sklearn.preprocessing import MinMaxScaler


# 3. Написать функцию нормализации данных (алгоритм нормализации взять согласно варианту, примеры реализации в списке
# источников). Для нормализации допускается использование готовых функций из сторонних библиотек.
def run_task_three():
    data = np.loadtxt("data/kc_house_data.csv", delimiter=',', skiprows=1)
    data = np.array(data, dtype=np.float32)
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)

    # Сохранение нормализованные данные в файл "normalized_data.csv"
    with open('data/normalized_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(normalized_data)


def run_task_four():
    data = np.loadtxt("data/normalized_data.csv", delimiter=',')
    data = np.array(data, dtype=np.float32)

    for i in range(data.shape[1]):
        column = data[:, i]
        normalized_column = (column - np.mean(column)) / np.std(column)
        data[:, i] = normalized_column

        # Рассчитать и вывести минимальное, максимальное и среднее значения
        print(f"Столбец {i + 1}")
        print(f"Минимальное значение: {np.min(normalized_column)}")
        print(f"Максимальное значение: {np.max(normalized_column)}")
        print(f"Среднее значение: {np.mean(normalized_column)}\n")


def run_task_five_one(initial_data, normalized_data):
    global actual, predicted

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

    return np.sqrt(np.mean((actual - predicted) ** 2))


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
    initial_data = np.loadtxt("data/kc_house_data.csv", delimiter=',', skiprows=1)
    initial_data = np.array(initial_data, dtype=np.float32)

    normalized_data = np.loadtxt("data/normalized_data.csv", delimiter=',')
    normalized_data = np.array(normalized_data, dtype=np.float32)

    rmse = run_task_five_one(initial_data, normalized_data)
    print(f"Результат функции расчёта ошибки RMSE: {rmse}")

    gradient_array = run_task_five_two(initial_data, normalized_data)
    print(f"Результат функции расчёта градиента ошибки: {gradient_array}")


def run_task_six():
    pass


if __name__ == '__main__':
    run_task_three()
    run_task_four()
    run_task_five()
    run_task_six()
