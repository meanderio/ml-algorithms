import numpy as np

class Softmax:
    def __init__(self, ):
        pass
    
    def forward(self, X):
        if X.ndim == 1:
            x_shifted = X - np.max(X)
            exp_x = np.exp(x_shifted)
            self.output = exp_x / np.sum(exp_x)
        else:
            x_shifted = X - np.max(X, axis=1, keepdims=True)
            exp_x = np.exp(x_shifted)
            self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backprop(self, y):
        pass