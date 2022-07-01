from cProfile import label
from matplotlib import markers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



irises = pd.read_csv('iris.csv', header=None, encoding='utf-8')

X = np.array(irises.iloc[1:100, [2, 4]])

y = np.where(irises.iloc[1:100, -1] == 'Iris-setosa', 1, -1)

plt.scatter(X[:50, 0], X[:50, 1], color='blue', marker='s', label='Щетинистый')
plt.scatter(X[50:100, 0], X[50:100, 1], color='green', marker='o', label='Разноцветный')
plt.xlabel('Длина чашедистика [см]')
plt.ylabel('Длина лепестка [см]')
plt.legend(loc='upper left')



plt.show()