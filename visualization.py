import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def plot_boundary_solutions(X, y, classifier, resolution=0.2, x_label='', y_label='', title=''):
    markers = ('s', 'x', 'o')
    colors = ('red', 'blue', 'lightgreen', 'gray')

    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = min(X[:][0]) - 2, max(X[:][0]) + 2
    x2_min, x2_max = min(X[:][1]) - 2, max(X[:][1]) + 2

    x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                   np.arange(x2_min, x2_max, resolution))

    y_pred = np.array(classifier.predict(np.array([x1_mesh.ravel(), x2_mesh.ravel()]).transpose().tolist())).reshape(x1_mesh.shape)
    with open("perc.mdl", "w") as file:
        for i in range(len(y_pred)):
            for j in range(len(y_pred[0])):
                file.write(str(y_pred[i][j]) + " ")
    with open("X1.mdl", "w") as file:
        for i in range(len(x1_mesh)):
            for j in range(len(x1_mesh[0])):
                file.write(str(x1_mesh[i][j]) + " ")
    with open("X2.mdl", "w") as file:
        for i in range(len(x2_mesh)):
            for j in range(len(x2_mesh[0])):
                file.write(str(x2_mesh[i][j]) + " ")
    plt.contourf(x1_mesh, x2_mesh, y_pred, cmap=cmap, alpha=0.3)
    plt.xlim(x1_mesh.min(), x1_mesh.max())
    plt.ylim(x2_mesh.min(), x2_mesh.max())
    y = np.array(y)
    X = np.array(X)
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    color=colors[idx], alpha=0.8,
                    label=cl, marker=markers[idx], edgecolor='black')

    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
   # plt.show()


def plot_graph(x, y, x_label, y_label, tittle):
    plt.plot(x, y)
    plt.grid(visible=True)
    plt.title(tittle)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plor_gr(X, y, resolution=0.2):
    #markers = ('s', 'x', 'o')
    colors = ('red', 'blue', 'lightgreen', 'gray')

    cmap = ListedColormap(colors[:len(np.unique(y))])
    X[0], X[1] = X[0][:-1], X[1][:-1]
    X[0] = [float(x) for x in X[0]]
    X[1] = [float(x) for x in X[1]]
    x1_min, x1_max = min(X[0]) - 2, max(X[0]) + 2
    x2_min, x2_max = min(X[1]) - 2, max(X[1]) + 2
    x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                   np.arange(x2_min, x2_max, resolution))
    plt.contourf(X[0], X[1], np.array(y).reshape(x1_mesh.shape), cmap=cmap, alpha=0.3)
    plt.xlim(X[0].min(), X[0].max())
    plt.ylim(X[1].min(), X[1].max())
    plt.legend()
    plt.show()
