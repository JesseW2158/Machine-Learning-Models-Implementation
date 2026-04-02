import numpy as np

class Regression:
    def __init__(self, loss_function, n_iterations=1000, learning_rate=0.01):
        self.loss_function = loss_function
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    # Xavier initialization for weights
    def initalize_weights(self, n_features):
        limit = 1 / np.sqrt(n_features)
        self.weights = np.random.uniform(-limit, limit, n_features)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.initalize_weights(n_features)

        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights)
            loss = self.loss_function(y, y_predicted)
            gradient = -2 * np.dot(X.T, (y - y_predicted)) / n_samples
            self.weights -= self.learning_rate * gradient

class LinearRegression(Regression):
    def __init__(self, loss_function, n_iterations=1000, learning_rate=0.01, fit_intercept=True, normalize=False):
        super().__init__(loss_function, n_iterations, learning_rate)
        self.fit_intercept = fit_intercept
        self.normalize = normalize