import numpy as np

class DenseLayer:
    def __init__(self, m, n):
        self.weights = np.random.randn(m, n)
        self.biases  = np.zeros((1, n))
    
    def forward(self, X):
        self.output = X @ self.weights + self.biases

    def backprop(self, y):
        pass