# Библиотека для работы с csv файлом
import pandas as pd
# Библиотека для работы с матрицами
import numpy as np
# Библиотека для нормализация данных MinMax
from sklearn.preprocessing import MinMaxScaler
# Библиотека для построения графиков
import matplotlib.pyplot as plt

# Задание 3. Функция для нормализации данных MinMax всех четырёх столбцов (price, bedrooms, sqft_living, waterfront).
# Далее с сохранением в файл "normalized_data.csv"
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


# Задание 5.1 Функция для расчёта ошибки RMSE
def rmse_error_calculation_func(normalized_data, prediction_data):
    mse = ((normalized_data - prediction_data) ** 2).mean()
    return np.sqrt(mse)

# Задание 5.2 Функция расчёта градиента функции ошибки RMSE
def calculating_gradient_rmse_error_func(normalized_data, prediction_data):
    gradient = 2 * (prediction_data - normalized_data) / len(normalized_data)
    return gradient


# Задание 6. Функция градиентного спуска для коэффициентов линейной регрессии.
def gradient_descent_linear_regression_coefficients_func(x, y, learning_rate=0.01, number_iterations=1000):
    theta = np.zeros(x)
    for i in range(number_iterations):
        grad = calculating_gradient_rmse_error_func(y, x)
        theta = theta - learning_rate * grad
    return theta


# Задание 7. Функция линейной зависимости, коэффициенты инициализированы случайным образом
def linear_regression(x, y, learning_rate=0.01, number_iterations=1000):
    m, n = x.shape
    odds = np.random.randn(n, 1)
    x = np.hstack((np.ones((m, 1)), x))
    J_history = []
    for i in range(number_iterations):
        print("Сгенерированные коэффициенты линейной регрессии: " + str(odds))
        h = x.dot(odds)
        loss = h - y
        J = 1/(2*m) * np.sum(loss**2)
        gradient = x.T.dot(loss) / m
        odds -= learning_rate * gradient
        J_history.append(J)
    return odds, J_history

# Задание 8. Отобразить ошибку обученной модели с помощью написанной функции ошибки.
def display_trained_model_error_func():
    normalized_data = pd.read_csv('data/normalized_data.csv', sep=',', skiprows=[1])
    prediction_data = pd.read_csv('data/prediction_data.csv', sep=',', skiprows=[1])
    rmse = rmse_error_calculation_func(normalized_data, prediction_data)

    print(f"rmse = :{rmse.to_string()}")


# Задание 9. Вывести 3 двумерных графика зависимости у от каждого х.
# На графиках точками отобразить тестовые данные и линией срез обученной регрессии
def display_dependency_graph_func(normalized_data):
    # Строим графики зависимости у (price) от х1, х2, x3 (bedrooms, sqft_living, waterfront)
    for i in range(normalized_data.shape[1]):
        plt.plot(normalized_data[:, 0], normalized_data[:, 1], 'ro')
        plt.plot(x2seq, lin_func(x1seq, x2seq), 'r')
    plt.ylabel("y стоимость квартиры")
    plt.xlabel("x количество спален")
    plt.show()

    for i in range(normalized_data.shape[1]):
        plt.plot(normalized_data[:, 0], normalized_data[:, 2], 'ro')
        plt.plot(x2seq, lin_func(x1seq, x2seq), 'r')
    plt.ylabel("y стоимость квартиры")
    plt.xlabel("x квадратные метры")
    plt.show()

    for i in range(normalized_data.shape[1]):
        plt.plot(normalized_data[:, 0], normalized_data[:, 3], 'ro')
        plt.plot(x2seq, lin_func(x1seq, x2seq), 'r')
    plt.ylabel("y стоимость квартиры")
    plt.xlabel("x квартира на побережье")
    plt.show()

if __name__ == '__main__':
    min_max_data_normalize_func()
    calculate_mathematical_operations_func()

    data = np.array(pd.read_csv('data/normalized_data.csv', sep=','))

    column_x = np.array(data[:, 0])
    column_y = np.array(data[:, 1])
    gradient_descent = gradient_descent_linear_regression_coefficients_func(column_x, column_y)

