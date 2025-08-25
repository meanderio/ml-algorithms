import numpy as np

import sys


class NeuralNetwork:
    def __init__(self, layers=[], alpha=0.1, n_iters=10_000):
        self.layers     = layers
        self.n_layers   = len(layers) - 1 # don't count input
        self.alpha      = alpha
        self.n_iters    = n_iters
        self.weights    = self._initialize_weights()
        self.biases     = self._initialize_biases()
        self.costs      = []
        self.n_samples  = 0
        self.n_features = 0

    def debug(self,):
        print("layers")
        print(self.layers)

        print("weights")
        for W in self.weights: print(W.shape)

        print("biases")
        for b in self.biases: print(b.shape)

    def _initialize_weights(self,):
        weights = []
        for i in range(1, len(self.layers)):
            W = np.random.randn(self.layers[i], self.layers[i-1])
            weights.append(W)
        return weights
    
    def _initialize_biases(self,):
        biases = []
        for i in range(1, len(self.layers)):
            b = np.random.randn(self.layers[i],1)
            biases.append(b)
        return biases

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        y = y.reshape(1, self.n_samples)
        for _ in range(self.n_iters):
            # feed forward to get y_pred
            y_hat, A = self.feed_forward(X)
            # calculate cost
            cost  = self.calculate_cost(y_hat, y)
            # backprop to update weights
            dws, dbs = self.back_prop(X, y_hat, y, A)
            # update the weights and biases
            self.gradient_descent(dws, dbs)
        return None

    def gradient_descent(self, dws, dbs):
        for i in range(self.n_layers):
            self.weights[i] -= self.alpha * dws[i]
            self.biases[i]  -= self.alpha * dbs[i]

    def _binary_cross_entropy(self, y_hat, y):
        epsilon = 1e-10
        return - ((y * np.log(y_hat + epsilon)) + (1 + y) * np.log(1 - y_hat + epsilon))

    def calculate_cost(self, y_hat, y):
        m    = y.shape[1]
        loss = self._binary_cross_entropy(y_hat, y)
        cost = (1 / m) * np.sum(loss, axis=1)
        self.costs.append(np.sum(cost))
        return self.costs[-1]

    def feed_forward(self, X):
        z = X.T
        A = []
        for i in range(self.n_layers):
            z = self.weights[i] @ z + self.biases[i]
            z = self.activation(z); A.append(z)
        return A[-1], A

    def activation(self, z):
        return self._sigmoid(z)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def back_prop(self, X, y_hat, y, A):
        dws, dbs = [0] * self.n_layers, [0] * self.n_layers
        m  = y_hat.shape[1]
        dw = ((1/m) * (A[-1] - y)) @ A[-1-1].T
        db = np.sum((1/m) * (A[-1] - y), keepdims=True)
        dz = self.weights[-1].T @ ((1/m) * (A[-1] - y))

        dws[-1] = dw; dbs[-1] = db

        for i in range(self.n_layers-2,0,-1):
            dw = (dz * (A[i] * (1 - A[i]))) @ A[i-1].T
            db = np.sum(dw, axis=1, keepdims=True)
            dz = self.weights[i].T @ (dz * (A[i] * (1 - A[i])))
            dws[i] = dw
            dbs[i] = db

        dw = (dz * (A[0] * (1 - A[0]))) @ X
        db = np.sum(dw, axis=1, keepdims=True)
        dws[0] = dw; dbs[0] = db

        return dws, dbs

    def predict_proba(self, X):
        y_hat, _ = self.feed_forward(X)
        return y_hat

    def predict(self, X):
        return np.where(self.predict_proba(X) > 0.5, 1, 0)