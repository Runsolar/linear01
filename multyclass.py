import pandas as pd

import visualization
from perceptron02 import Perceptron
from data_preparation import train_test_split
from visualization import plot_boundary_solutions
from itertools import combinations

irises = pd.read_csv('iris.csv', header=None, encoding='utf-8')

y = [(irises.iloc[1:51, -1]).tolist(),
     (irises.iloc[51:101, -1]).tolist(),
     (irises.iloc[101:, -1]).tolist()]

x = [(irises.iloc[1:51, [2, 4]]).values.tolist(),
     (irises.iloc[51:101, [2, 4]]).values.tolist(),
     (irises.iloc[101:, [2, 4]]).values.tolist()]

objs = []

comb = list(combinations(range(len(y)), 2))


def most_frequent(list):
    return max(set(list), key=list.count)


for i in range(len(comb)):
    print("{}. {} и {}".format(i+1, y[comb[i][0]][0], y[comb[i][1]][0]))
data_list = []

for i in range(len(comb)):
    objs.append(Perceptron(eta0=0.1, random_state=1))
    x_comb = []
    y_comb = []
    for j in (comb[i]):
        x_comb += x[j]
        y_comb += y[j]
    y_comb_res = [1 if y_comb[a] == y_comb[0] else -1 for a in range(len(y_comb))]
    X_train, y_train, X_test, y_test = train_test_split(x_comb, y_comb_res, 0.3, True)
    objs[i].fit(X_train, y_train)
    plot_boundary_solutions(X_test, y_test, objs[i], 0.02, "признак1", "признак2", "Граница решений")

    with open("perc.mdl", "r") as file:
        data = file.read().replace('-1', y_comb[-1]).replace('1', y_comb[0]).split(" ")
    data_list.append(data)


frequent = []
min_len = len(data_list[0])
for i in range(len(data_list)):
    min_len = min(min_len, len(data_list[i]))


for j in range(min_len):
    s = [data_list[i][j] for i in range(len(data_list))]
    frequent.append(most_frequent(s))

for i in range(len(frequent)):
    if frequent[i] == "Iris-setosa":
        frequent[i] = 0
    elif frequent[i] == "Iris-versicolor":
        frequent[i] = 1
    else:
        frequent[i] = 2


with open("X1.mdl", "r") as file:
    data_x1 = file.read().split(" ")
with open("X2.mdl", "r") as file:
    data_x2 = file.read().split(" ")

visualization.plor_gr([data_x1, data_x2], frequent)
