
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class Perceptron(object):

    def __init__(self, n_iter = 10, eta = 0.01, random_state=1):
        self.n_iter = n_iter
        self.eta = eta
        self.random_state = random_state
        self.w_arr = []

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
        
        self.model_save()

        return self
    
    def net_input(self, x):
        return np.dot(x, self.w[1:]) + self.w[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)

    def model_save(self):
        ind = self.errors.index(min(self.errors))
        w_ = str(self.w_arr[ind])
        with open("perceptron01.mdl", "w") as file:
            file.write(w_)

    def model_load(self):
        with open("perceptron01.mdl", "r") as file:
            res = file.readline()
        return res
