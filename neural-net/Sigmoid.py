import numpy as np

class Sigmoid:
    def __init__(self, ):
        pass
    
    def forward(self, X):
        self.output = 1 / (1 + np.exp(-X))
    
    def backprop(self, y):
        pass