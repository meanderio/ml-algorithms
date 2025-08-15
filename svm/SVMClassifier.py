import numpy as np
import sys

class SVMClassifier:
    def __init__(self, lr=0.001, lambda_param=0.001, n_iters=5000):
        self.lr           = lr
        self.n_iters      = n_iters
        self.lambda_param = lambda_param
        self.w            = None
        self.b            = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iters):
            for i, x in enumerate(X):
                cnd     = y[i] * (np.dot(X[i], self.w) - self.b) < 1
                self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(y[i],X[i]) * cnd)
                self.b -= self.lr * (y[i] * cnd)
    
    def predict_prob(self, X):
        return np.dot(X, self.w) - self.b
    
    def predict(self, X):
        return np.sign(self.predict_prob(X))

