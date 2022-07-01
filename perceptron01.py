
import numpy as np
import pandas as pd

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
    
    def net_input(self, x):
        return np.dot(x, self.w[1:]) + self.w[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)

if __name__ == "__main__":

    irises = pd.read_csv('iris.csv', header=None, encoding='utf-8')

    X = np.array(irises.iloc[1:100, [2, 4]])

    y = np.where(irises.iloc[1:100, -1] == 'Iris-setosa', 1, -1)

    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.3, True)

    obj1  = Perceptron()

    obj1.fit(X, y)

    exmaple = X[60, :]

    print(obj1.predict(exmaple))


    print(obj1.errors)


    
