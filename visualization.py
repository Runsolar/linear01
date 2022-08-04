import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def plot_boundary_solutions(X, y, classifier, resolution=0.2, x_label='', y_label='', title=''):
    markers = ('s', 'x', 'o')
    colors = ('red', 'blue', 'lightgreen', 'gray')

    cmap = ListedColormap(colors[:len(np.unique(y))])

    X = np.array(X)
    x1_min, x1_max = min(X[:, 0]) - 2, max(X[:, 0]) + 2
    x2_min, x2_max = min(X[:, 1]) - 2, max(X[:, 1]) + 2

    x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                   np.arange(x2_min, x2_max, resolution))

    y_pred = np.array(classifier.predict(np.array([x1_mesh.ravel(), x2_mesh.ravel()]).transpose().tolist())).reshape(x1_mesh.shape)

    plt.contourf(x1_mesh, x2_mesh, y_pred, cmap=cmap, alpha=0.3)
    plt.xlim(x1_mesh.min(), x1_mesh.max())
    plt.ylim(x2_mesh.min(), x2_mesh.max())
    y = np.array(y)

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    color=colors[idx], alpha=0.8,
                    label=cl, marker=markers[idx], edgecolor='black')

    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_graph(x, y, x_label, y_label, tittle):
    plt.plot(x, y)
    plt.grid(visible=True)
    plt.title(tittle)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_gr(X, y, frequent, x1, y1, resolution=0.2):
    markers = ('s', 'x', 'o', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray')

    cmap = ListedColormap(colors[:len(np.unique(frequent))])

    X = np.array(X)
    x1_min, x1_max = min(X[:, 0]) - 2, max(X[:, 0]) + 2
    x2_min, x2_max = min(X[:, 1]) - 2, max(X[:, 1]) + 2

    x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, (x1_max - x1_min)/200),
                                   np.arange(x2_min, x2_max, (x2_max - x2_min)/200))
    plt.contourf(x1_mesh, x2_mesh, np.array(frequent[:-1]).reshape(x1_mesh.shape), cmap=cmap, alpha=0.5)

    y = np.array(y)
    y1 = np.array(y1)
    x1 = np.array(x1)

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    color=colors[idx], alpha=0.8,
                    label=cl, marker=markers[idx], edgecolor='black')
        plt.scatter(x=x1[y1 == cl, 0],
                    y=x1[y1 == cl, 1],
                    color=colors[idx], alpha=0.8,
                    marker=markers[idx], edgecolor='black')
    plt.scatter(x=x1[:, 0],
                y=x1[:, 1], facecolors='none',
                alpha=1.0, linewidths=1, s=100,
                label='Тестовый набор', marker='o', edgecolor='black')

    plt.xlim(x1_mesh.min(), x1_mesh.max())
    plt.ylim(x2_mesh.min(), x2_mesh.max())
    plt.legend()
    plt.show()


def multy_predict(X, classifier, num=200):
    X = np.array(X)
    x1_min, x1_max = min(X[:, 0]) - 2, max(X[:, 0]) + 2
    x2_min, x2_max = min(X[:, 1]) - 2, max(X[:, 1]) + 2

    x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, (x1_max - x1_min)/num),
                                   np.arange(x2_min, x2_max, (x2_max - x2_min)/num))

    y_pred = np.array(classifier.predict(np.array([x1_mesh.ravel(),
                                                   x2_mesh.ravel()]).transpose().tolist())).reshape(x1_mesh.shape)

    with open("perc.mdl", "w") as file:
        for i in range(len(y_pred)):
            for j in range(len(y_pred[0])):
                file.write(str(y_pred[i][j]) + " ")

