import numpy as np

class Tanh:
    def __init__(self, ):
        pass
    
    def forward(self, X):
        self.output = (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))
    
    def backprop(self, y):
        pass