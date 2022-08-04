import pandas as pd
import visualization
from perceptron02 import Perceptron
from data_preparation import train_test_split
from visualization import plot_boundary_solutions
from itertools import combinations


irises = pd.read_csv('iris.csv', header=None, encoding='utf-8')

# данные для обучения и теста 3 нейронов
y = [(irises.iloc[1:51, -1]).tolist(),
     (irises.iloc[51:101, -1]).tolist(),
     (irises.iloc[101:, -1]).tolist()]

x = [(irises.iloc[1:51, [2, 4]]).values.tolist(),
     (irises.iloc[51:101, [2, 4]]).values.tolist(),
     (irises.iloc[101:, [2, 4]]).values.tolist()]


# функция определения наиболее часто встреченного элемента + количество одинаковых в консоль (спорные точки)
disputed_points = 0
def most_frequent(lst):
    global disputed_points
    counter = 0
    res = lst[0]
    disp = []
    for el in lst:
        curr_frequency = lst.count(el)
        if curr_frequency > counter:
            counter = curr_frequency
            res = el
        disp.append(curr_frequency)
    if disp.count(max(disp)) == 3:
        disputed_points += 1
    return res


# количество ошибок классификации
def errors_count(x, y, classifier):
    y_pred = classifier.predict(x)
    errors_num = 0
    for i in range(len(y)):
        if y_pred[i][0] != y[i]:
            errors_num += 1
    return errors_num


# комбинации индексов классов
comb = list(combinations(range(len(y)), 2))

# вывод порядка пар
for i in range(len(comb)):
    print("{}. {} и {}".format(i+1, y[comb[i][0]][0], y[comb[i][1]][0]))

# данные для трех выходных сеток
X = (irises.iloc[1:, [2, 4]]).values.tolist()
y_all = (irises.iloc[1:, -1]).values.tolist()

# приведение меток к числовому виду для классификации
for i in range(len(y_all)):
    if y_all[i] == "Iris-setosa":
        y_all[i] = 0
    elif y_all[i] == "Iris-versicolor":
        y_all[i] = 1
    else:
        y_all[i] = 2

# данные для трех выходных сеток
X_train, y_train, X_test, y_test = train_test_split(X, y_all, 0.3, True)

objs = []  # список обученных нейронов
data_list = []  # результаты классификации
err_train = 0  # суммарное количество ошибок классификации на тренировочной выборке
err_test = 0  # суммарное количество ошибок классификации на тестовой выборке

# попарное обучений нейронов
for i in range(len(comb)):
    objs.append(Perceptron(eta0=0.02, random_state=1))
    x_comb = []
    y_comb = []
    for j in (comb[i]):
        x_comb += x[j]
        y_comb += y[j]
    y_comb_res = [1 if y_comb[a] == y_comb[0] else -1 for a in range(len(y_comb))]
    X_train1, y_train1, X_test1, y_test1 = train_test_split(x_comb, y_comb_res, 0.3, True)  # данные для попарной классификации
    objs[i].fit(X_train1, y_train1)
    plot_boundary_solutions(X_test1, y_test1, objs[i], 0.2, "признак1", "признак2", "Граница решений")

    visualization.multy_predict(X_train, objs[i])  # классификация для 3 классов
    err_train += errors_count(X_train1, y_train1, objs[i])  # оценка точности классификатора на тренировочной выборке
    err_test += errors_count(X_test1, y_test1, objs[i])  # оценка точности классификатора на тренировочной выборке
    with open("perc.mdl", "r") as file:
        data = file.read().replace('-1', y_comb[-1]).replace('1', y_comb[0]).split(" ")
    data_list.append(data)

print("Оценка точности: ", 1 - err_train/len(X_train))  # или
print(1-err_test/len(X_test))

# определение границы решения для 3 классов
frequent = []

# результат классификации 3 классов
for j in range(len(data_list[0])):
    s = [data_list[i][j] for i in range(len(data_list))]
    frequent.append(most_frequent(s))
print("Количество спорных точек: ", disputed_points)

# приведение меток к числовому виду для визуализации
for i in range(len(frequent)):
    if frequent[i] == "Iris-setosa":
        frequent[i] = 0
    elif frequent[i] == "Iris-versicolor":
        frequent[i] = 1
    else:
        frequent[i] = 2
# Построение границы решений для трех классов
visualization.plot_gr(X_train, y_train, frequent, x1=X_test, y1=y_test)

