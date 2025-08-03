import numpy as np

class LinearRegresssion:
    
    def __init__(self, lr=0.10, n_iter=1_000):
        self._lr    = lr
        self._iters = n_iter
        self._w     = None
        self._b     = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._w = np.zeros(n_features)
        self._b = 0

        for _ in range(self._iters):
            y_pred = X @ self._w + self._b

            dw = -(2 / n_samples) * X.T  @ (y - y_pred)
            db = -(2 / n_samples) * np.sum(y - y_pred)

            self._w -= self._lr * dw
            self._b -= self._lr * db

    def predict(self, X):
        return X @ self._w + self._b 
