
import pandas as pd
from perceptron02 import Perceptron
from data_preparation import train_test_split
from visualization import plot_boundary_solutions, plot_graph


irises = pd.read_csv('iris.csv', header=None, encoding='utf-8')

X = (irises.iloc[1:100, [2, 4]]).values.tolist()

y = (irises.iloc[1:100, -1])
y = y.where(y == 'Iris-setosa', -1)
y = y.where(y != 'Iris-setosa', 1)
y = y.values.tolist()

X_train, y_train, X_test, y_test = train_test_split(X, y, 0.3, True)

obj1 = Perceptron(eta0=0.1, random_state=1)

obj1.fit(X_train, y_train)

example = X_train

print(obj1.predict(example))

print(obj1.errors)

plot_graph(range(1, len(obj1.errors) + 1), obj1.errors, 'Эпохи', 'Ошибка', 'Количество ошибок классификации')

plot_boundary_solutions(X_test, y_test, obj1, 0.02, "признак1", "признак2", "Граница решений")