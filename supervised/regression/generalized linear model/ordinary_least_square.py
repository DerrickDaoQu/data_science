import numpy as np
from sklearn.datasets.samples_generator import make_regression

# uncomment the following line to cross check with sklearn results
# from sklearn.linear_model import LinearRegression


# linear regression using gradient descent
class OrdinaryLeastSquare:
    def __init__(self):
        self.intercept = None
        self.coef = None

    # calculate model residuals
    def __resid(self, y_true, y):
        return y_true - y

    # train model
    # parameters: x: feature matrix, y: target variable, e: termination criteria
    # learning_rate: model update step, verbose: whether to print debug info
    # fit_intercept: whether to fit intercept
    def fit(self, x, y, max_iter=1000, e=None, learning_rate=0.001, verbose=False, fit_intercept=True):
        n = x.shape[0]

        # append ones or zeros at the first column depending on whether to fit intercept
        X = np.append(np.ones((n, 1)), x, axis=1) if fit_intercept else np.append(np.zeros((n, 1)), x, axis=1)
        d = X.shape[1]

        # initialize coefficients
        coef = np.random.randn(d)

        costs = []
        for i in range(max_iter):

            # generate fitted values
            fitted_values = X.dot(coef.T).reshape(-1, 1)

            # calculate model residuals
            resid = self.__resid(y, fitted_values).reshape(-1, 1)

            # calculate and record model cost
            cost = np.mean(resid ** 2)

            if verbose:
                print('{}: {}', i + 1, cost)
            costs.append(cost)

            # update coefficients other than the intercept
            coef[1:] = coef[1:] + learning_rate * np.mean(np.multiply(resid, x), axis=0)

            # update the intercept coefficient if fit_intercept is True
            if fit_intercept:
                coef[0] += learning_rate * np.mean(resid)

            # early termination if cost delta is less than threshold e
            if i > 0 and e is not None and np.abs((cost - costs[i - 1])) / costs[i - 1] <= e:
                if verbose:
                    print('converged at iteration {}', i)
                break

        # record intercept and coefficients
        self.intercept = coef[0]
        self.coef = coef[1:]

        # return self class for chaining call
        return self


if __name__ == '__main__':
    # make data
    x, y = make_regression(1000, 10, 10)
    y = y.reshape(-1, 1)

    # normalize and standardize
    x = (x - x.mean(axis=0)) / x.std(axis=0)

    ols = OrdinaryLeastSquare()
    o = ols.fit(x, y, max_iter=10000, verbose=False)
    print(o.intercept, ', ', o.coef)

    # uncomment the following block to cross check with sklearn results
    # lm = LinearRegression()
    # lm.fit(x, y)
    # print(lm.intercept_, ', ', lm.coef_)
