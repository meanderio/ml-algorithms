import numpy as np

class Sigmoid:
    def __init__(self, ):
        self.output = None
    
    def forward(self, X):
        X_clamped = np.clip(X, -50, 50)
        self.output = 1.0 / (1.0 + np.exp(-X_clamped))
        return self.output
    
    def backward(self, dY):
        return dY * self.output * (1.0 - self.output)
    
    def step(self,):
        pass