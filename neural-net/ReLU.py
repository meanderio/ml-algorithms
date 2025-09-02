import numpy as np

class ReLU:
    def __init__(self, ):
        self._mask = None
    
    def forward(self, X):
        self._mask = X > 0
        return np.where(self._mask, X, 0.0)
    
    def backward(self, dY):
       return dY * self._mask
    
    def step(self,):
        pass