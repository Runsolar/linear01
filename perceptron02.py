import math01 as m1
import random


class Perceptron(object):

    def __init__(self, n_iter=10, eta0=0.01, random_state=1):
        self.i = 0
        self.n_iter = n_iter
        self.eta = eta0
        self.random_state = random_state
        self.w_arr = []

    def fit(self, X, y):

        self.errors = []
        self.w = [random.gauss(-0.2, 0.1) for _ in range(m1.shape(X)[1] + 1)]

        for _ in range(self.n_iter):
            error_ = 0.0
            for x, target in zip(X, y):
                update = self.eta * (target - self.predict(x))
                w_update = m1.dot(update, x)
                for i in range(len(self.w[1:])-1):
                    self.w[i+1] += w_update[i]
                self.w[0] += update

                error_ += int(update != 0.0)
            self.w_arr.append(self.w)
            self.errors.append(error_)

        self.model_save()

        return self

    def net_input(self, x):
        d = m1.dot(x, self.w[1:])
        if type(d) == list:
            return m1.summ(m1.dot(x, self.w[1:]), self.w[0])
        else:
            return m1.dot(x, self.w[1:]) + self.w[0]

    def predict(self, x):
        return m1.where(self.net_input(x), 1, -1)

    def model_save(self):
        ind = self.errors.index(min(self.errors))
        w_ = str(self.w_arr[ind])
        with open("perceptron01.mdl", "w") as file:
            file.write(w_)

    def model_load(self):
        with open("perceptron01.mdl", "r") as file:
            res = file.readline()[1:-1].split(', ')
        return res
