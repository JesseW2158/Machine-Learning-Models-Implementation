class Regression():


class LinearRegression(Regression):
    def __init__(self, n_iterations=1000, learning_rate=0.01, fit_intercept=True, normalize=False, loss_function):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.loss_function = loss_function