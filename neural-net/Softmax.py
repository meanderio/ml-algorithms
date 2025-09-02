import numpy as np

class Softmax:
    def __init__(self, ):
        self.output = None
    
    def forward(self, X):
        self.inputs = X
        if X.ndim == 1:
            x_shifted = X - np.max(X)
            exp_x = np.exp(x_shifted)
            self.output = exp_x / np.sum(exp_x)
        else:
            x_shifted = X - np.max(X, axis=1, keepdims=True)
            exp_x = np.exp(x_shifted)
            self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output
    
    def backward(self, dY):
        dot = np.sum(dY * self.output, axis=1, keepdims=True)
        print(self.output.shape, dY.shape, dot.shape)
        return self.output * (dY - dot)
    
    def step(self,):
        pass