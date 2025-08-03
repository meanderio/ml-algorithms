import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.1, n_iters=1_000):
        self._lr = 0.1
        self._iters = n_iters
        self._w = None
        self._b = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._w = np.zeros(n_features)
        self._b = 0

        for _ in range(self._iters):
            y_pred = self._sigmoid(np.dot(X, self._w) + self._b)

            dw = (2 / n_samples) * X.T @ (y_pred - y)
            db = (2 / n_samples) * np.sum(y_pred - y)

            self._w -= self._lr * dw
            self._b -= self._lr * db
    
    def predict_prob(self, X):
        return self._sigmoid(np.dot(X, self._w) + self._b)
    
    def predict(self, X):
        return np.where(self.predict_prob(X) > 0.5, 1, 0)
