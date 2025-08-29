import numpy as np

class ReLU:
    def __init__(self, ):
        pass
    
    def forward(self, X):
        self.output = np.maximum(0, X)
    
    def backprop(self, y):
        pass