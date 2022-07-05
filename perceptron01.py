
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class Perceptron(object):

    def __init__(self, n_iter = 10, eta = 0.01, random_state=1):
        self.n_iter = n_iter
        self.eta = eta
        self.random_state = random_state

    def fit(self, X, y):

        self.rgen = np.random.RandomState(self.random_state)

        self.errors = []
        self.w = self.rgen.normal(loc = 0., scale = 0.1, size = X.shape[1] + 1)

        for _ in range(self.n_iter):

            error_ = 0.0
            for x, target in zip(X, y):

                update = self.eta*(target - self.predict(x))
                
                self.w[1:] += update * x
                self.w[0] += update

                error_ += int(update != 0.0)

                #print(self.w)

            self.errors.append(error_)

        return self
    
    def net_input(self, x):
        return np.dot(x, self.w[1:]) + self.w[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)


def train_test_split(x_input, y_input, test_percent, mixing):
    x_train, x_test, y_train, y_test = [], [], [], []

    train_percent = 1 - test_percent

    if mixing:

        mixed = list(zip(x_input, y_input))
        random.shuffle(mixed)

        mixed_train = mixed[:int(train_percent*len(mixed))]
        mixed_test = mixed[int(train_percent*len(mixed)):]

        for val1, val2 in zip(mixed_train, mixed_test):

            x_train.append(val1[0])
            y_train.append(val1[1])
            x_test.append(val2[0])
            y_test.append(val2[1])

    else:

        x_train, x_test = x_input[:int(train_percent*len(x_input))], x_input[int(train_percent*len(x_input)):]
        y_train, y_test = y_input[:int(train_percent*len(y_input))], y_input[int(train_percent*len(y_input)):]

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def plot_boundary_solutions(X, y, resolution, perc):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                   np.arange(x2_min, x2_max, resolution))
    
    y_pred = perc.predict(np.array([x1_mesh.ravel(), x2_mesh.ravel()]).transpose()).reshape(x1_mesh.shape)

    plt.contourf(x1_mesh, x2_mesh, y_pred, colors=['red', 'blue'], alpha=0.5)

    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', alpha=0.5, label='1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', alpha=0.5, label='-1')

    plt.xlim(x1_mesh.min(), x1_mesh.max())
    plt.ylim(x2_mesh.min(), x2_mesh.max())

    plt.legend()

    plt.show()


if __name__ == "__main__":

    irises = pd.read_csv('iris.csv', header=None, encoding='utf-8')

    X = np.array(irises.iloc[1:100, [2, 4]]) 

    X_train_setosa = np.array(irises.iloc[1:36, [2, 4]])

    X_train_versicolor = np.array(irises.iloc[51:86, [2, 4]])

    X_train_virginica = np.array(irises.iloc[101:136, [2, 4]])

    y = np.where(irises.iloc[1:100, -1] == 'Iris-setosa', 1, -1)

    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.3, True)

    obj1  = Perceptron()

    obj1.fit(X_train, y_train)

    example = X_train

    print(obj1.predict(example))

    print(obj1.errors)

    plt.plot(range(1, len(obj1.errors) + 1) , obj1.errors)
    plt.title('Количество ошибок классификации')
    plt.grid(True)
    plt.show()

    plot_boundary_solutions(X_test, y_test, 0.02, obj1)
