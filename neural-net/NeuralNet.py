import numpy as np
np.random.seed(42)
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

    def _rmse(self, y_hat, y):
        return np.sqrt(np.mean((y_hat - y) ** 2))
    
    def _binary_cross_entropy(self, y_hat, y):
        epsilon = 1e-10
        return - np.sum((((y * np.log(y_hat + epsilon)) + (1 + y) * np.log(1 - y_hat + epsilon))))

    def calculate_cost(self, y_hat, y):
        m    = y.shape[1]
        loss = self._rmse(y_hat, y)
        cost = (1 / m) * loss
        self.costs.append(np.sum(cost))
        return self.costs[-1]

    def feed_forward(self, X):
        z = X.T
        A = []
        for i in range(self.n_layers):
            z = self.weights[i] @ z + self.biases[i]
            z = self.activation(z)
            #print(z)
            if (i+1 == len(self.layers)) and (self.layers[i] > 1): z = z.argmax(0)
            A.append(z)
        return A[-1], A

    def activation(self, z):
        return self._leaky_relu(z)
        #return self._tanh(z)
        #return self._sigmoid(z)

    def _leaky_relu(self, z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)
    
    def _tanh(self, z):
        e_pos = np.exp(z)
        e_neg = np.exp(-z)
        return (e_pos - e_neg) / (e_pos + e_neg)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def back_prop(self, X, y_hat, y, A):
        A.insert(0, X.T)
        dws, dbs = [0] * self.n_layers, [0] * self.n_layers
        m        = y_hat.shape[1]
        dz       = 1
        for i in range(self.n_layers-1,-1,-1):
            # back prop errors
            da = dz * ((1/m) * (A[i+1] - y))
            dw = da @ A[i].T
            db = np.sum(da, keepdims=True)
            # add weight and biases updates
            dws[i] = dw; dbs[i] = db
            # calculate propagator
            dz = self.weights[i].T @ da
        return dws, dbs

    def predict_proba(self, X):
        y_hat, _ = self.feed_forward(X)
        return y_hat

    def predict(self, X):
        y_hat = self.predict_proba(X)
        if self.layers[-1] == 1:
            return np.where(y_hat > 0.5, 1, 0)
        print(y_hat)
        return y_hat.argmax(0)