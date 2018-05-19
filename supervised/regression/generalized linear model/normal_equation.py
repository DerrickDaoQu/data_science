import numpy as np
from sklearn.datasets.samples_generator import make_regression


# uncomment the following line to cross check with sklearn results
# from sklearn.linear_model import LinearRegression

# linear regression using normal equation
class NormalEquation:

    def __init__(self):
        self.intercept = None
        self.coef = None

    # calculate model residuals
    def __resid(self, y_true, y):
        return y_true - y

    # train linear regression model using normal equation with pseudo-inverse
    # parameters: x: feature matrix, y: target variable, fit_intercept: whether to fit intercept
    def fit(self, x, y, fit_intercept=True):
        n = x.shape[0]

        # append ones or zeros at the first column depending on whether to fit intercept
        X = np.append(np.ones((n, 1)), x, axis=1) if fit_intercept else np.append(np.zeros((n, 1)), x, axis=1)

        # calculate coefficients using pseudo-inverse
        coef = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

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

    normal_equation = NormalEquation()
    o = normal_equation.fit(x, y)
    print(o.intercept, ', ', o.coef)

    # uncomment the following block to cross check with sklearn results
    # lm = LinearRegression()
    # lm.fit(x, y)
    # print(lm.intercept_, ', ', lm.coef_)
