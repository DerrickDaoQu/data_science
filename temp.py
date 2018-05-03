import numpy as np
import pandas as pd
import random
# import sklearn
# from sklearn.datasets.samples_generator import make_regression
import pylab
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt


def compute_cost_function(m, t0, t1, x, y):
    return 1 / 2 / m * sum([(t0 + t1 * np.asarray([x[i]]) - y[i]) ** 2 for i in range(m)])


def gradient_descent(alpha, x, y, ep=0.0001, max_iter=1500):
    converged = False
    iter = 0
    m = x.shape[0]  # number of samples

    # initial theta
    t0 = 0
    t1 = 0

    # total error, J(theta)
    J = compute_cost_function(m, t0, t1, x, y)
    # Iterate Loop
    num_iter = 0
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        grad0 = 1.0 / m * sum([(t0 + t1 * np.asarray([x[i]]) - y[i]) for i in range(m)])
        grad1 = 1.0 / m * sum([(t0 + t1 * np.asarray([x[i]]) - y[i]) * np.asarray([x[i]]) for i in range(m)])

        # update the theta_temp
        temp0 = t0 - alpha * grad0
        temp1 = t1 - alpha * grad1

        # update theta
        t0 = temp0
        t1 = temp1

        # mean squared error
        e = compute_cost_function(m, t0, t1, x, y)
        J = e  # update error
        iter += 1  # update iter

        if iter == max_iter:
            print('Max interactions exceeded!')
            converged = True

    return t0, t1


def plot_cost_function(x, y, m):
    t0 = list(range(0, x.shape[0]))
    j_values = []
    for i in range(len(t0)):
        j_values.append(compute_cost_function(m, i, i, x, y)[0])
    print('j_values', len(j_values), len(x), len(y))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, j_values, label='parametric curve')
    ax.legend()
    plt.show()


x_data = np.arange(1000).reshape(-1, 1)
y = (10 * x_data[:, 0] + 2000).reshape(-1, 1)
t0, t1 = gradient_descent(0.000001, x_data, y, max_iter=100)