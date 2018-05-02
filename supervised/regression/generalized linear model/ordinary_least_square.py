import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_regression


# from sklearn.linear_model import LinearRegression


class OrdinaryLeastSquare:
    def __init__(self):
        self.intercept = None
        self.coef = None

    def __resid(self, y_true, y):
        return y_true - y

    def fit(self, x, y, max_iter=1000, e=None, learning_rate=0.001, verbose=False, fit_intercept=True):
        n = x.shape[0]
        X = np.append(np.ones((n, 1)), x, axis=1) if fit_intercept else np.append(np.zeros((n, 1)), x, axis=1)
        d = X.shape[1]
        coef = np.random.randn(d)

        costs = []
        for i in range(max_iter):
            fitted_values = X.dot(coef.T).reshape(-1, 1)
            resid = self.__resid(y, fitted_values).reshape(-1, 1)
            cost = np.sqrt(np.mean(resid ** 2))
            if verbose:
                print('{}: {}', i + 1, cost)
            costs.append(cost)

            coef[1:] = coef[1:] + learning_rate * np.mean(np.multiply(resid, x), axis=0)
            if fit_intercept:
                coef[0] += learning_rate * np.mean(resid)

            if i > 0 and e is not None and np.abs((cost - costs[i - 1])) / costs[i - 1] <= e:
                if verbose:
                    print('converged at iteration {}', i)
                break

        self.intercept = coef[0]
        self.coef = coef[1:]
        return self


if __name__ == '__main__':
    x_data, y = make_regression(1000, 10, 10)
    x_data = (x_data - x_data.mean(axis=0)) / x_data.std(axis=0)
    # lm = LinearRegression()
    # lm.fit(x_data, y)
    # print(lm.intercept_, ', ', lm.coef_)
    ols = OrdinaryLeastSquare()
    o = ols.fit(x_data, y.reshape(-1, 1), max_iter=10000, verbose=False)
    print(o.intercept, ', ', o.coef)
